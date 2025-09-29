// ogc_physics.cu
// Minimal CUDA implementation of Offset Geometric Contact (OGC)
// Based on: "Offset Geometric Contact", TOG 44(4), Aug 2025
// Implements: OGC contact sets (V-F and E-E), 2-stage activation, per-vertex trust regions,
// and a VBD-like implicit step (inertia + stretch + contacts).
//
// Build: nvcc -O3 -std=c++17 ogc_physics.cu -o ogc_physics
//
// This is a teaching/starting point implementation with clear structure and heavy comments.
// It uses a naive broadphase (O(N^2)) for clarity; replace with BVH/grid for scale.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <string>
#include <cassert>

// ---------- CUDA helpers ----------
#define CUDA_OK(stmt) do { cudaError_t err = (stmt); if (err != cudaSuccess) { \
  fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1);} } while(0)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------- float2/3/4 helpers ----------
static __host__ __device__ inline float3 makef3(float x,float y,float z){ float3 r; r.x=x; r.y=y; r.z=z; return r; }
static __host__ __device__ inline float2 makef2(float x,float y){ float2 r; r.x=x; r.y=y; return r; }

static __host__ __device__ inline float3 operator+(const float3&a,const float3&b){ return makef3(a.x+b.x,a.y+b.y,a.z+b.z); }
static __host__ __device__ inline float3 operator-(const float3&a,const float3&b){ return makef3(a.x-b.x,a.y-b.y,a.z-b.z); }
static __host__ __device__ inline float3 operator*(const float3&a,float s){ return makef3(a.x*s,a.y*s,a.z*s); }
static __host__ __device__ inline float3 operator*(float s,const float3&a){ return a*s; }
static __host__ __device__ inline float3 operator/(const float3&a,float s){ float inv=1.0f/s; return a*inv; }
static __host__ __device__ inline float  dot(const float3&a,const float3&b){ return a.x*b.x+a.y*b.y+a.z*b.z; }
static __host__ __device__ inline float3 cross(const float3&a,const float3&b){
  return makef3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
static __host__ __device__ inline float  len2(const float3&a){ return dot(a,a); }
static __host__ __device__ inline float  len (const float3&a){ return sqrtf(len2(a)); }
static __host__ __device__ inline float3 normalize(const float3&a){ float l = len(a); return (l>0)? a/l : makef3(0,0,0); }

// Atomic add for float3 via float atomics
static __device__ inline void atomicAddFloat3(float3* dst, const float3& v){
  atomicAdd(&dst->x, v.x);
  atomicAdd(&dst->y, v.y);
  atomicAdd(&dst->z, v.z);
}

// atomicMin for float via CAS (for positive floats)
static __device__ inline void atomicMinFloat(float* addr, float val){
  int* addr_as_i = (int*)addr;
  int old = *addr_as_i, assumed;
  do {
    assumed = old;
    float old_f = __int_as_float(assumed);
    float min_f = fminf(old_f, val);
    old = atomicCAS(addr_as_i, assumed, __float_as_int(min_f));
  } while(assumed != old);
}

// ---------- Mesh data ----------
struct Tri { int v0, v1, v2; };
struct Edge { int v0, v1; int t0, t1; int opp0, opp1; }; // neighbors for feasible checks

// per-triangle local edge indices -> global edge id
struct TriEdges { int e01, e12, e20; };

// CSR adjacency: ranges into a flat array
struct CSR {
  int* offsets;   // size = count + 1
  int* indices;   // flat list
};

struct DeviceMesh {
  int nV, nT, nE;

  // geometry & state
  float3* x;         // positions (current)
  float3* xPrev;     // positions at last contact detection (X_prev)
  float3* v;         // velocities
  float*  m;         // masses

  Tri*  tris;        // size nT
  Edge* edges;       // size nE
  TriEdges* triEdges;// size nT

  // adjacency
  CSR v2n; // vertex -> neighbor vertices (star)
  CSR v2t; // vertex -> incident triangles
  CSR v2e; // vertex -> incident edges

  // edge rest length for stretch springs
  float*  edgeRest;  // size nE
};

// ---------- OGC contact structures ----------
enum FaceType : int { FACE_TRI_INTERIOR = 0, FACE_EDGE = 1, FACE_VERTEX = 2 };

// a vertex-facet contact from vertex side; if type==EDGE, sub=local edge (0,1,2); if type==VERTEX, sub=local vertex (0,1,2)
struct VFContact { int tri; int type; int sub; };

// an edge-edge contact (from edge e to other edge eOther); type==0: true edge-edge; type==1: closest near endpoint (vertex id recorded)
struct EEContact { int eOther; int type; int vEndpoint; };

// capacities (tune as needed)
#ifndef MAX_VF_CONTACTS
#define MAX_VF_CONTACTS 64
#endif
#ifndef MAX_EE_CONTACTS
#define MAX_EE_CONTACTS 16
#endif

// contact lists
struct ContactBuffers {
  // VF
  int* vfCount;                  // nV
  VFContact* vfList;             // nV * MAX_VF_CONTACTS

  // EE
  int* eeCount;                  // nE
  EEContact* eeList;             // nE * MAX_EE_CONTACTS
};

// ---------- Per-iteration buffers ----------
struct IterBuffers {
  // distances for conservative bounds
  float* dmin_v; // nV
  float* dmin_t; // nT
  float* dmin_e; // nE

  float* b;      // nV (bounds)

  // target inertia
  float3* Y;     // nV

  // solver accumulators
  float3* f;     // nV force-like residual
  float*  H;     // nV diagonal preconditioner

  // truncation counter
  int* numExceed; // single int on device
};

// ---------- Simulation parameters ----------
struct SimParams {
  float dt;            // timestep
  float3 aext;         // external acceleration (e.g., gravity)
  float kc;            // collision stiffness (quadratic stage)
  float r;             // contact radius
  float gamma_p;       // bound relaxation (Eq.21), e.g., 0.45
  float gamma_e;       // fraction threshold to redo detection
  float spring_k;      // stretch spring stiffness
  // friction parameters (extension points)
  float mu;            // Coulomb coefficient (not used in this minimal build)
  float eps_v;         // friction regularizer
};

// ---------- Closest point to triangle result ----------
struct ClosestTriResult {
  float3 c;     // closest point
  float d;      // distance
  int type;     // FACE_TRI_INTERIOR / FACE_EDGE / FACE_VERTEX
  int sub;      // for edge: 0->(v0,v1),1->(v1,v2),2->(v2,v0); for vertex: 0/1/2
  float w0,w1,w2; // barycentric of c wrt tri
};

// Robust closest point to triangle with classification (Ericson)
static __device__ inline ClosestTriResult closestPointTriangle(const float3& p, const float3& a, const float3& b, const float3& c){
  ClosestTriResult R{};
  const float3 ab = b - a, ac = c - a, ap = p - a;
  float d1 = dot(ab, ap);
  float d2 = dot(ac, ap);
  if (d1 <= 0.f && d2 <= 0.f) { // barycentric (1,0,0)
    R.c = a; R.w0=1; R.w1=0; R.w2=0; R.type = FACE_VERTEX; R.sub=0;
    R.d = len(p - R.c); return R;
  }
  const float3 bp = p - b;
  float d3 = dot(ab, bp);
  float d4 = dot(ac, bp);
  if (d3 >= 0.f && d4 <= d3) { // (0,1,0)
    R.c = b; R.w0=0; R.w1=1; R.w2=0; R.type = FACE_VERTEX; R.sub=1;
    R.d = len(p - R.c); return R;
  }
  float vc = d1*d4 - d3*d2;
  if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f){
    float v = d1 / (d1 - d3);
    R.c = a + ab * v; R.w0=1.f - v; R.w1=v; R.w2=0; R.type = FACE_EDGE; R.sub=0; // edge (a,b)
    R.d = len(p - R.c); return R;
  }
  const float3 cp = p - c;
  float d5 = dot(ab, cp);
  float d6 = dot(ac, cp);
  if (d6 >= 0.f && d5 <= d6) { // (0,0,1)
    R.c = c; R.w0=0; R.w1=0; R.w2=1; R.type = FACE_VERTEX; R.sub=2;
    R.d = len(p - R.c); return R;
  }
  float vb = d5*d2 - d1*d6;
  if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f){
    float w = d2 / (d2 - d6);
    R.c = a + ac * w; R.w0=1.f - w; R.w1=0; R.w2=w; R.type = FACE_EDGE; R.sub=2; // edge (c,a) -> sub=2
    R.d = len(p - R.c); return R;
  }
  float va = d3*d6 - d5*d4;
  if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f){
    float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    R.c = b + (c - b) * w; R.w0=0; R.w1=1.f - w; R.w2=w; R.type = FACE_EDGE; R.sub=1; // edge (b,c) -> sub=1
    R.d = len(p - R.c); return R;
  }
  // inside face
  float denom = 1.0f / (va + vb + vc);
  float v = vb * denom;
  float w = vc * denom;
  R.w0 = 1.f - v - w; R.w1 = v; R.w2 = w;
  R.c = a * R.w0 + b * R.w1 + c * R.w2;
  R.type = FACE_TRI_INTERIOR; R.sub = -1;
  R.d = len(p - R.c);
  return R;
}

// Projection foot of x3 onto line (x1,x2)
static __device__ inline float3 footOnLine(const float3& x1, const float3& x2, const float3& x3){
  float3 d = x2 - x1; float L2 = len2(d);
  if (L2 <= 0.f) return x1;
  float t = dot(x3 - x1, d) / L2;
  return x1 + d * t;
}

// Segment-segment closest points (returns s,t in [0,1], points c1, c2, and distance)
static __device__ inline float segSegClosest(const float3& p1, const float3& q1, const float3& p2, const float3& q2,
                                            float& s, float& t, float3& c1, float3& c2)
{
  const float3 d1 = q1 - p1; // segment 1
  const float3 d2 = q2 - p2; // segment 2
  const float3 r  = p1 - p2;
  const float a = dot(d1,d1); // squared length segment 1
  const float e = dot(d2,d2); // squared length segment 2
  const float f = dot(d2,r);

  float EPS=1e-12f;
  if (a <= EPS && e <= EPS) { s = t = 0.f; c1 = p1; c2 = p2; return len(c1 - c2); }
  if (a <= EPS) { s = 0.f; t = f / e; t = fmaxf(0.f,fminf(1.f,t)); }
  else {
    float c = dot(d1,r);
    if (e <= EPS) { t = 0.f; s = fmaxf(0.f, fminf(1.f, -c / a)); }
    else {
      float b = dot(d1,d2);
      float denom = a*e - b*b;
      if (denom != 0.f) s = fmaxf(0.f, fminf(1.f, (b*f - c*e)/denom)); else s = 0.f;
      t = (b*s + f)/e;
      if (t < 0.f){ t = 0.f; s = fmaxf(0.f, fminf(1.f, -c/a)); }
      else if (t > 1.f){ t = 1.f; s = fmaxf(0.f, fminf(1.f, (b - c)/a)); }
    }
  }
  c1 = p1 + d1 * s;
  c2 = p2 + d2 * t;
  return len(c1 - c2);
}

// ---------- Two-stage activation (Eq. 18) and derivative ----------
// We choose tau = r/2 and k_c' = k_c * (r^2 / 4) to make C2 stitching cleanly (derivation in assistant notes).
static __host__ __device__ inline float activation_dEd_d(const float d, const float r, const float kc){
  if (d >= r) return 0.f; // inactive
  const float tau = 0.5f * r;
  const float kcp = kc * (r*r * 0.25f); // k_c' = kc * r^2 / 4
  if (d >= tau){
    // g = (kc/2)*(r - d)^2 -> g'(d) = -kc*(r - d)
    return -kc * (r - d);
  }else{
    // g = -kcp*log(d) + b -> g'(d) = -kcp/d
    float dd = fmaxf(d, 1e-6f);
    return -(kcp / dd);
  }
}

// ---------- Feasible region checks for blocks (Eqs. 8, 9, 15) ----------

static __device__ inline bool checkVertexFeasibleRegion(const DeviceMesh& M, int vIdx, const float3& xq, float r){
  const float3 xv = M.x[vIdx];
  float3 dv = xq - xv;
  if (len(dv) > r + 1e-6f) return false;
  const int start = M.v2n.offsets[vIdx];
  const int end   = M.v2n.offsets[vIdx+1];
  for (int it=start; it<end; ++it){
    int vnb = M.v2n.indices[it];
    float3 nplane = xv - M.x[vnb]; // (xv - x_v')
    if (dot(dv, nplane) < -1e-8f) return false;
  }
  return true;
}

static __device__ inline bool checkEdgeFeasibleRegion(const DeviceMesh& M, int eIdx, const float3& xq, float r){
  const Edge e = M.edges[eIdx];
  const float3 x1 = M.x[e.v0], x2 = M.x[e.v1];
  // radial distance
  // (distance <= r) guaranteed by contact detection; check axial span and face-cuts
  if (dot(xq - x1, x2 - x1) <= 0.f) return false;
  if (dot(xq - x2, x1 - x2) <= 0.f) return false;

  // cuts by neighbor faces (opp vertices)
  if (e.opp0 >= 0){
    float3 p = footOnLine(x1, x2, M.x[e.opp0]);
    if (dot(xq - p, p - M.x[e.opp0]) < -1e-8f) return false;
  }
  if (e.opp1 >= 0){
    float3 p = footOnLine(x1, x2, M.x[e.opp1]);
    if (dot(xq - p, p - M.x[e.opp1]) < -1e-8f) return false;
  }
  return true;
}

// ---------- Contact detection (naive broadphase) ----------

__global__ void resetDMin(float* arr, int n, float rq){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i<n) arr[i] = rq;
}

__global__ void resetCounts(int* cnt, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) cnt[i]=0; }

// Per-vertex VF detection (Algorithm 1, simplified: naive search over all triangles)
__global__ void detectVF(const DeviceMesh M, const float r, const float rq,
                         IterBuffers B, ContactBuffers C)
{
  int v = blockIdx.x*blockDim.x + threadIdx.x;
  if (v >= M.nV) return;
  float3 xv = M.x[v];
  float dmin = rq;

  // Avoid duplicates per vertex by storing keys (type,id)
  int localN = 0;
  VFContact localList[MAX_VF_CONTACTS];

  // Helper to test if tri contains v
  auto triHasVertex = [&] __device__ (const Tri& t, int vi){
    return (t.v0==vi || t.v1==vi || t.v2==vi);
  };

  for (int tIdx=0; tIdx<M.nT; ++tIdx){
    Tri t = M.tris[tIdx];
    if (triHasVertex(t, v)) continue;

    float3 a = M.x[t.v0], b = M.x[t.v1], c = M.x[t.v2];
    ClosestTriResult R = closestPointTriangle(xv, a, b, c);
    dmin = fminf(dmin, R.d);
    atomicMinFloat(&B.dmin_t[tIdx], R.d);

    if (R.d < r){
      // Determine contact face 'a'
      int type = R.type;
      int sub  = R.sub;

      bool accept = false;
      int globalID = -1; // for dedup: tri/edge/vertex id
      if (type == FACE_TRI_INTERIOR){
        accept = true; globalID = tIdx;
      } else if (type == FACE_EDGE){
        // global edge id from tri + local edge
        TriEdges te = M.triEdges[tIdx];
        int ge = (sub==0)? te.e01 : (sub==1? te.e12 : te.e20);
        if (ge >= 0 && checkEdgeFeasibleRegion(M, ge, xv, r)){
          accept = true; globalID = ge;
        }
      } else { // FACE_VERTEX
        int lv = sub;
        int gv = (lv==0? t.v0 : (lv==1? t.v1 : t.v2));
        if (checkVertexFeasibleRegion(M, gv, xv, r)){
          accept = true; globalID = gv;
        }
      }

      if (accept){
        // dedup (local)
        bool dup=false;
        for (int k=0;k<localN;++k){
          const VFContact& L = localList[k];
          if (L.type==type){
            if (type==FACE_TRI_INTERIOR && L.tri==tIdx) { dup=true; break; }
            if (type==FACE_EDGE){
              TriEdges te2 = M.triEdges[L.tri];
              int ge2 = (L.sub==0? te2.e01 : (L.sub==1? te2.e12 : te2.e20));
              if (ge2==globalID){ dup=true; break; }
            }
            if (type==FACE_VERTEX){
              Tri tt = M.tris[L.tri];
              int gv2 = (L.sub==0? tt.v0 : (L.sub==1? tt.v1 : tt.v2));
              if (gv2==globalID){ dup=true; break; }
            }
          }
        }
        if (!dup && localN < MAX_VF_CONTACTS){
          VFContact c; c.tri = tIdx; c.type=type; c.sub=sub;
          localList[localN++] = c;
        }
      }
    }
  }

  B.dmin_v[v] = dmin;

  // write to global
  int base = v * MAX_VF_CONTACTS;
  int n = localN;
  C.vfCount[v] = n;
  for (int i=0;i<n;++i) C.vfList[base+i] = localList[i];
}

// Per-edge EE detection (Algorithm 2, simplified: naive all-pairs)
__global__ void detectEE(const DeviceMesh M, const float r, const float rq,
                         IterBuffers B, ContactBuffers C)
{
  int e = blockIdx.x*blockDim.x + threadIdx.x;
  if (e >= M.nE) return;
  Edge E = M.edges[e];
  float3 p1 = M.x[E.v0], q1 = M.x[E.v1];
  float dmin = rq;

  int localN=0;
  EEContact localList[MAX_EE_CONTACTS];

  // helper: share a vertex?
  auto share = [&] __device__ (const Edge& a, const Edge& b){
    return (a.v0==b.v0 || a.v0==b.v1 || a.v1==b.v0 || a.v1==b.v1);
  };

  for (int j=0;j<M.nE;++j){
    if (j==e) continue;
    Edge F = M.edges[j];
    if (share(E,F)) continue;

    float3 p2 = M.x[F.v0], q2 = M.x[F.v1];
    float s=0.f,t=0.f; float3 c1,c2;
    float d = segSegClosest(p1,q1,p2,q2,s,t,c1,c2);
    dmin = fminf(dmin, d);

    if (d < r){
      // classify: if c1 near endpoint -> treat as vertex contact in edge-only manifold
      int type = 0, vend = -1;
      if (s < 1e-3f) { type=1; vend=E.v0; }
      else if (s > 1.f-1e-3f) { type=1; vend=E.v1; }
      if (type==1){
        if (!checkVertexFeasibleRegion(M, vend, c1, r)) continue;
      }
      if (localN < MAX_EE_CONTACTS){
        EEContact ec; ec.eOther=j; ec.type=type; ec.vEndpoint=vend;
        localList[localN++] = ec;
      }
    }
  }

  B.dmin_e[e] = dmin;
  int base = e * MAX_EE_CONTACTS;
  C.eeCount[e] = localN;
  for (int i=0;i<localN;++i) C.eeList[base+i] = localList[i];
}

// ---------- Conservative bounds (Eq. 21–27) ----------

__global__ void computeBounds(const DeviceMesh M, IterBuffers B, const SimParams P){
  int v = blockIdx.x*blockDim.x + threadIdx.x;
  if (v >= M.nV) return;

  float dmin_v = B.dmin_v[v];

  // dEmin_v = min over incident edges' dmin_e
  float dEmin = 1e30f;
  int sE = M.v2e.offsets[v], eE = M.v2e.offsets[v+1];
  for (int it=sE; it<eE; ++it){
    int eIdx = M.v2e.indices[it];
    dEmin = fminf(dEmin, B.dmin_e[eIdx]);
  }

  // dTmin_v = min over incident triangles' dmin_t
  float dTmin = 1e30f;
  int sT = M.v2t.offsets[v], eT = M.v2t.offsets[v+1];
  for (int it=sT; it<eT; ++it){
    int t = M.v2t.indices[it];
    dTmin = fminf(dTmin, B.dmin_t[t]);
  }

  float dmin_all = fminf(dmin_v, fminf(dEmin, dTmin));
  dmin_all = fmaxf(dmin_all, 0.0f);
  B.b[v] = P.gamma_p * dmin_all;
}

// ---------- Initialization and inertia ----------
__global__ void buildY(const DeviceMesh M, const SimParams P, float3* Y){
  int v = blockIdx.x*blockDim.x + threadIdx.x;
  if (v >= M.nV) return;
  Y[v] = M.x[v] + M.v[v]*P.dt + P.aext*(P.dt*P.dt);
}

__global__ void applyInitialGuessTruncated(const DeviceMesh M, const float3* Y, IterBuffers B){
  int v = blockIdx.x*blockDim.x + threadIdx.x;
  if (v >= M.nV) return;
  float3 xinit = Y[v];
  float3 xprev = M.xPrev[v];
  float3 d = xinit - xprev;
  float L = len(d);
  float bv = B.b[v];
  float3 xstar = (L <= bv || L <= 1e-12f) ? xinit : (xprev + d*(bv/L));
  // write into current positions:
  ((float3*)M.x)[v] = xstar;
}

// ---------- Solver accumulators ----------

__global__ void zeroFH(float3* f, float* H, int nV){
  int v = blockIdx.x*blockDim.x + threadIdx.x;
  if (v < nV){ f[v] = makef3(0,0,0); H[v] = 0.f; }
}

// inertia contribution: f += -m/h^2 (x - Y); H += m/h^2
__global__ void addInertia(const DeviceMesh M, const float3* Y, IterBuffers B, const SimParams P){
  int v = blockIdx.x*blockDim.x + threadIdx.x;
  if (v >= M.nV) return;
  float invh2 = 1.f / (P.dt*P.dt);
  float w = M.m[v] * invh2;
  float3 r = M.x[v] - Y[v];
  atomicAddFloat3(&B.f[v], makef3(-w*r.x, -w*r.y, -w*r.z));
  atomicAdd(&B.H[v], w);
}

// stretch springs on edges: E=0.5*k*(|d|-L)^2 (symmetric)
__global__ void addStretch(const DeviceMesh M, IterBuffers B, const SimParams P){
  int e = blockIdx.x*blockDim.x + threadIdx.x;
  if (e >= M.nE) return;
  Edge E = M.edges[e];
  int i = E.v0, j = E.v1;
  float3 xi = M.x[i], xj = M.x[j];
  float3 d = xi - xj; float L = len(d);
  float L0 = M.edgeRest[e];
  float3 dir = (L>1e-9f)? (d / L) : makef3(0,0,0);
  float k = P.spring_k;
  float coeff = k * (1.f - (L0 / fmaxf(L, 1e-9f))); // gradient magnitude
  float3 fi = dir * coeff; // force on i (positive along dir)
  float3 fj = fi * (-1.f);

  atomicAddFloat3(&B.f[i], fi);
  atomicAddFloat3(&B.f[j], fj);
  // crude diagonal preconditioner
  atomicAdd(&B.H[i], k);
  atomicAdd(&B.H[j], k);
}

// ---------- Contact application ----------

// apply VF contacts gathered per-vertex; symmetric reactions to tri vertices
__global__ void applyVFContacts(const DeviceMesh M, const ContactBuffers C, IterBuffers B, const SimParams P){
  int v = blockIdx.x*blockDim.x + threadIdx.x;
  if (v >= M.nV) return;
  int n = C.vfCount[v];
  int base = v * MAX_VF_CONTACTS;
  float3 xv = M.x[v];

  for (int k=0;k<n;++k){
    VFContact c = C.vfList[base+k];
    Tri t = M.tris[c.tri];
    float3 a = M.x[t.v0], b = M.x[t.v1], cc = M.x[t.v2];

    float3 Cp; float d; float3 nrm;
    float w0=0,w1=0,w2=0;
    if (c.type == FACE_TRI_INTERIOR){
      ClosestTriResult R = closestPointTriangle(xv, a,b,cc);
      Cp = R.c; d = R.d; nrm = normalize(xv - Cp); w0=R.w0; w1=R.w1; w2=R.w2;
    } else if (c.type == FACE_EDGE){
      int ge = (c.sub==0? M.triEdges[c.tri].e01 : (c.sub==1? M.triEdges[c.tri].e12 : M.triEdges[c.tri].e20));
      Edge E = M.edges[ge];
      float s,t; float3 c1,c2;
      d = segSegClosest(xv, xv, M.x[E.v0], M.x[E.v1], s,t, c1,c2); // degenerate seg for point
      Cp = c2;
      nrm = normalize(xv - Cp);
      // distribute to edge endpoints
      w0 = 0; w1 = 0; w2 = 0;
    } else { // FACE_VERTEX
      int gv = (c.sub==0? t.v0 : (c.sub==1? t.v1 : t.v2));
      Cp = M.x[gv]; d = len(xv - Cp); nrm = normalize(xv - Cp);
      w0=w1=w2=0;
    }

    float dEd = activation_dEd_d(d, P.r, P.kc); // g'(d)
    float Fn = -dEd;                            // repulsion magnitude

    float3 F = nrm * Fn;                        // on v
    atomicAddFloat3(&B.f[v], F);
    atomicAdd(&B.H[v], P.kc);                   // diagonal approx

    // Reaction distribution:
    if (c.type == FACE_TRI_INTERIOR){
      float3 Fr = F * (-1.f);
      atomicAddFloat3(&B.f[t.v0], Fr * w0);
      atomicAddFloat3(&B.f[t.v1], Fr * w1);
      atomicAddFloat3(&B.f[t.v2], Fr * w2);
      atomicAdd(&B.H[t.v0], P.kc*0.3333f);
      atomicAdd(&B.H[t.v1], P.kc*0.3333f);
      atomicAdd(&B.H[t.v2], P.kc*0.3333f);
    } else if (c.type == FACE_EDGE){
      int ge = (c.sub==0? M.triEdges[c.tri].e01 : (c.sub==1? M.triEdges[c.tri].e12 : M.triEdges[c.tri].e20));
      Edge E = M.edges[ge];
      // parameter t along edge for distribution
      float3 evec = M.x[E.v1] - M.x[E.v0];
      float L2 = len2(evec);
      float tpar = (L2>0.f)? dot(Cp - M.x[E.v0], evec)/L2 : 0.f;
      tpar = fmaxf(0.f,fminf(1.f,tpar));
      float3 Fr = F * (-1.f);
      atomicAddFloat3(&B.f[E.v0], Fr * (1.f - tpar));
      atomicAddFloat3(&B.f[E.v1], Fr * tpar);
      atomicAdd(&B.H[E.v0], P.kc*0.5f);
      atomicAdd(&B.H[E.v1], P.kc*0.5f);
    } else { // FACE_VERTEX
      int gv = (c.sub==0? t.v0 : (c.sub==1? t.v1 : t.v2));
      atomicAddFloat3(&B.f[gv], F * (-1.f));
      atomicAdd(&B.H[gv], P.kc);
    }

    // TODO (friction, lagged): add tangential force capped at mu*Fn using stored directions from previous iteration.
  }
}

// apply EE contacts; symmetric reactions on both edges
__global__ void applyEEContacts(const DeviceMesh M, const ContactBuffers C, IterBuffers B, const SimParams P){
  int e = blockIdx.x*blockDim.x + threadIdx.x;
  if (e >= M.nE) return;
  int n = C.eeCount[e];
  int base = e * MAX_EE_CONTACTS;

  Edge E = M.edges[e];
  float3 p1 = M.x[E.v0], q1 = M.x[E.v1];

  for (int k=0;k<n;++k){
    EEContact ce = C.eeList[base+k];
    Edge F = M.edges[ce.eOther];
    float3 p2 = M.x[F.v0], q2 = M.x[F.v1];

    float s,t; float3 c1,c2;
    float d = segSegClosest(p1,q1,p2,q2,s,t, c1,c2);
    float3 nrm = normalize(c1 - c2);
    float dEd = activation_dEd_d(d, P.r, P.kc);
    float Fn = -dEd;
    float3 F12 = nrm * Fn; // acts to separate (on e push +n, on eOther push -n)

    // distribute along edges
    float wi0 = (1.f - s), wi1 = s;
    float wj0 = (1.f - t), wj1 = t;

    // on edge e
    atomicAddFloat3(&B.f[E.v0], F12 * wi0);
    atomicAddFloat3(&B.f[E.v1], F12 * wi1);
    atomicAdd(&B.H[E.v0], P.kc*0.5f);
    atomicAdd(&B.H[E.v1], P.kc*0.5f);

    // reaction on edge eOther
    float3 Fr = F12 * (-1.f);
    atomicAddFloat3(&B.f[F.v0], Fr * wj0);
    atomicAddFloat3(&B.f[F.v1], Fr * wj1);
    atomicAdd(&B.H[F.v0], P.kc*0.5f);
    atomicAdd(&B.H[F.v1], P.kc*0.5f);

    // TODO (friction): add capped tangential terms as extension point.
  }
}

// ---------- Position update (VBD-like block step) ----------
__global__ void updatePositions(DeviceMesh M, IterBuffers B){
  int v = blockIdx.x*blockDim.x + threadIdx.x;
  if (v >= M.nV) return;
  float H = fmaxf(B.H[v], 1e-9f);
  float3 dx = B.f[v] / H; // local block descent
  ((float3*)M.x)[v] = M.x[v] + dx;
}

// ---------- Trust region truncation and exceed count ----------
__global__ void truncateToBounds(const DeviceMesh M, IterBuffers B){
  int v = blockIdx.x*blockDim.x + threadIdx.x;
  if (v >= M.nV) return;
  float3 xprev = M.xPrev[v];
  float3 d = M.x[v] - xprev;
  float L = len(d);
  float b = B.b[v];
  if (L > b && L > 1e-12f){
    ((float3*)M.x)[v] = xprev + d * (b / L);
    atomicAdd(B.numExceed, 1);
  }
}

// ---------- Host-side simple mesh/edge builder (CPU) ----------
struct HostMesh {
  std::vector<float3> X;
  std::vector<float3> V;
  std::vector<float>  M;
  std::vector<Tri>    Tris;
};

static inline uint64_t keyEdge(int a, int b){ uint64_t A = (uint32_t)std::min(a,b); uint64_t B=(uint32_t)std::max(a,b); return (A<<32)|B; }

// Build edges and adjacency (CPU)
static void buildEdgesAndAdj(const HostMesh& H,
                             std::vector<Edge>& edges,
                             std::vector<TriEdges>& triEdges,
                             std::vector<int>& v2n_offsets, std::vector<int>& v2n_indices,
                             std::vector<int>& v2t_offsets, std::vector<int>& v2t_indices,
                             std::vector<int>& v2e_offsets, std::vector<int>& v2e_indices)
{
  const int nV = (int)H.X.size();
  const int nT = (int)H.Tris.size();

  std::unordered_map<uint64_t, int> eMap; eMap.reserve(nT*2);
  edges.clear(); triEdges.resize(nT);

  auto addEdge = [&](int a,int b,int tri, int opp) -> int {
    uint64_t k = keyEdge(a,b);
    auto it = eMap.find(k);
    if (it==eMap.end()){
      Edge E; E.v0=std::min(a,b); E.v1=std::max(a,b);
      E.t0=tri; E.t1=-1; E.opp0=opp; E.opp1=-1;
      int id = (int)edges.size();
      edges.push_back(E);
      eMap[k]=id;
      return id;
    } else {
      int id = it->second;
      Edge& E = edges[id];
      E.t1 = tri;
      E.opp1 = opp;
      return id;
    }
  };

  // tri edges and fill opp
  for (int t=0;t<nT;++t){
    Tri T = H.Tris[t];
    int e01 = addEdge(T.v0, T.v1, t, T.v2);
    int e12 = addEdge(T.v1, T.v2, t, T.v0);
    int e20 = addEdge(T.v2, T.v0, t, T.v1);
    TriEdges TE{e01,e12,e20};
    triEdges[t]=TE;
  }

  // vertex neighbors & v2t, v2e
  std::vector<std::vector<int>> VV(nV), VT(nV), VE(nV);
  for (int t=0;t<nT;++t){
    Tri T = H.Tris[t];
    VT[T.v0].push_back(t); VT[T.v1].push_back(t); VT[T.v2].push_back(t);
    VV[T.v0].push_back(T.v1); VV[T.v0].push_back(T.v2);
    VV[T.v1].push_back(T.v0); VV[T.v1].push_back(T.v2);
    VV[T.v2].push_back(T.v0); VV[T.v2].push_back(T.v1);
  }
  for (int e=0;e<(int)edges.size();++e){
    VE[edges[e].v0].push_back(e);
    VE[edges[e].v1].push_back(e);
  }
  auto uniqSort = [](std::vector<int>& v){ std::sort(v.begin(),v.end()); v.erase(std::unique(v.begin(),v.end()), v.end()); };

  v2n_offsets.resize(nV+1); v2t_offsets.resize(nV+1); v2e_offsets.resize(nV+1);
  int accN=0, accT=0, accE=0;
  for (int v=0; v<nV; ++v){
    uniqSort(VV[v]); uniqSort(VT[v]); uniqSort(VE[v]);
    v2n_offsets[v]=accN; accN += (int)VV[v].size();
    v2t_offsets[v]=accT; accT += (int)VT[v].size();
    v2e_offsets[v]=accE; accE += (int)VE[v].size();
  }
  v2n_offsets[nV]=accN; v2t_offsets[nV]=accT; v2e_offsets[nV]=accE;

  v2n_indices.resize(accN); v2t_indices.resize(accT); v2e_indices.resize(accE);
  for (int v=0; v<nV; ++v){
    int s=v2n_offsets[v];
    for (int i=0;i<(int)VV[v].size();++i) v2n_indices[s+i]=VV[v][i];
    s=v2t_offsets[v];
    for (int i=0;i<(int)VT[v].size();++i) v2t_indices[s+i]=VT[v][i];
    s=v2e_offsets[v];
    for (int i=0;i<(int)VE[v].size();++i) v2e_indices[s+i]=VE[v][i];
  }
}

// ---------- Device mesh upload ----------
static DeviceMesh uploadMesh(const HostMesh& H){
  DeviceMesh M{};
  M.nV = (int)H.X.size(); M.nT = (int)H.Tris.size();

  // Build edges & adjacency
  std::vector<Edge> edges;
  std::vector<TriEdges> triEdges;
  std::vector<int> v2n_off,v2n_idx, v2t_off,v2t_idx, v2e_off,v2e_idx;
  buildEdgesAndAdj(H, edges, triEdges, v2n_off,v2n_idx, v2t_off,v2t_idx, v2e_off,v2e_idx);
  M.nE = (int)edges.size();

  // allocate & copy
  CUDA_OK(cudaMalloc(&M.x,      M.nV*sizeof(float3)));
  CUDA_OK(cudaMalloc(&M.xPrev,  M.nV*sizeof(float3)));
  CUDA_OK(cudaMalloc(&M.v,      M.nV*sizeof(float3)));
  CUDA_OK(cudaMalloc(&M.m,      M.nV*sizeof(float)));

  CUDA_OK(cudaMalloc(&M.tris,    M.nT*sizeof(Tri)));
  CUDA_OK(cudaMalloc(&M.edges,   M.nE*sizeof(Edge)));
  CUDA_OK(cudaMalloc(&M.triEdges,M.nT*sizeof(TriEdges)));

  // adjacency
  CUDA_OK(cudaMalloc(&M.v2n.offsets, (M.nV+1)*sizeof(int)));
  CUDA_OK(cudaMalloc(&M.v2n.indices, v2n_idx.size()*sizeof(int)));
  CUDA_OK(cudaMalloc(&M.v2t.offsets, (M.nV+1)*sizeof(int)));
  CUDA_OK(cudaMalloc(&M.v2t.indices, v2t_idx.size()*sizeof(int)));
  CUDA_OK(cudaMalloc(&M.v2e.offsets, (M.nV+1)*sizeof(int)));
  CUDA_OK(cudaMalloc(&M.v2e.indices, v2e_idx.size()*sizeof(int)));

  // edge rest length
  std::vector<float> L0(edges.size());
  for (int i=0;i<(int)edges.size();++i){
    int a=edges[i].v0, b=edges[i].v1;
    float3 xa = H.X[a], xb = H.X[b];
    L0[i] = sqrtf((xa.x-xb.x)*(xa.x-xb.x)+(xa.y-xb.y)*(xa.y-xb.y)+(xa.z-xb.z)*(xa.z-xb.z));
  }
  CUDA_OK(cudaMalloc(&M.edgeRest, M.nE*sizeof(float)));

  // copy data
  CUDA_OK(cudaMemcpy(M.x,     H.X.data(),   M.nV*sizeof(float3), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.xPrev, H.X.data(),   M.nV*sizeof(float3), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.v,     H.V.data(),   M.nV*sizeof(float3), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.m,     H.M.data(),   M.nV*sizeof(float),  cudaMemcpyHostToDevice));

  CUDA_OK(cudaMemcpy(M.tris,    H.Tris.data(), M.nT*sizeof(Tri), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.edges,   edges.data(),  M.nE*sizeof(Edge), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.triEdges,triEdges.data(), M.nT*sizeof(TriEdges), cudaMemcpyHostToDevice));

  CUDA_OK(cudaMemcpy(M.v2n.offsets, v2n_off.data(), (M.nV+1)*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.v2n.indices, v2n_idx.data(), v2n_idx.size()*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.v2t.offsets, v2t_off.data(), (M.nV+1)*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.v2t.indices, v2t_idx.data(), v2t_idx.size()*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.v2e.offsets, v2e_off.data(), (M.nV+1)*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.v2e.indices, v2e_idx.data(), v2e_idx.size()*sizeof(int), cudaMemcpyHostToDevice));

  CUDA_OK(cudaMemcpy(M.edgeRest, L0.data(), M.nE*sizeof(float), cudaMemcpyHostToDevice));

  return M;
}

static void freeDeviceMesh(DeviceMesh& M){
  cudaFree(M.x); cudaFree(M.xPrev); cudaFree(M.v); cudaFree(M.m);
  cudaFree(M.tris); cudaFree(M.edges); cudaFree(M.triEdges);
  cudaFree(M.v2n.offsets); cudaFree(M.v2n.indices);
  cudaFree(M.v2t.offsets); cudaFree(M.v2t.indices);
  cudaFree(M.v2e.offsets); cudaFree(M.v2e.indices);
  cudaFree(M.edgeRest);
  M = DeviceMesh{};
}

// ---------- Buffers alloc ----------
static IterBuffers allocIterBuffers(const DeviceMesh& M){
  IterBuffers B{};
  CUDA_OK(cudaMalloc(&B.dmin_v, M.nV*sizeof(float)));
  CUDA_OK(cudaMalloc(&B.dmin_t, M.nT*sizeof(float)));
  CUDA_OK(cudaMalloc(&B.dmin_e, M.nE*sizeof(float)));
  CUDA_OK(cudaMalloc(&B.b,      M.nV*sizeof(float)));
  CUDA_OK(cudaMalloc(&B.Y,      M.nV*sizeof(float3)));
  CUDA_OK(cudaMalloc(&B.f,      M.nV*sizeof(float3)));
  CUDA_OK(cudaMalloc(&B.H,      M.nV*sizeof(float)));
  CUDA_OK(cudaMalloc(&B.numExceed, sizeof(int)));
  return B;
}
static void freeIterBuffers(IterBuffers& B){
  cudaFree(B.dmin_v); cudaFree(B.dmin_t); cudaFree(B.dmin_e);
  cudaFree(B.b); cudaFree(B.Y); cudaFree(B.f); cudaFree(B.H); cudaFree(B.numExceed);
  B = IterBuffers{};
}

static ContactBuffers allocContactBuffers(const DeviceMesh& M){
  ContactBuffers C{};
  CUDA_OK(cudaMalloc(&C.vfCount, M.nV*sizeof(int)));
  CUDA_OK(cudaMalloc(&C.vfList,  M.nV*MAX_VF_CONTACTS*sizeof(VFContact)));
  CUDA_OK(cudaMalloc(&C.eeCount, M.nE*sizeof(int)));
  CUDA_OK(cudaMalloc(&C.eeList,  M.nE*MAX_EE_CONTACTS*sizeof(EEContact)));
  return C;
}
static void freeContactBuffers(ContactBuffers& C){
  cudaFree(C.vfCount); cudaFree(C.vfList);
  cudaFree(C.eeCount); cudaFree(C.eeList);
  C = ContactBuffers{};
}

// ---------- Simulation step (Algorithm 3) ----------
static void step(DeviceMesh& M, IterBuffers& B, ContactBuffers& C, const SimParams& P, int nIter,
                 bool& collisionDetectionRequired)
{
  dim3 vgrid((M.nV+127)/128), vblock(128);
  dim3 tgrid((M.nT+127)/128), tblock(128);
  dim3 egrid((M.nE+127)/128), eblock(128);

  if (collisionDetectionRequired){
    // reset dmin_t upper bound to rq
    float rq = P.r + 2.0f * len(P.aext) * P.dt * P.dt + 2.0f * P.dt; // heuristic query radius
    resetDMin<<<tgrid,tblock>>>(B.dmin_t, M.nT, rq);
    CUDA_OK(cudaGetLastError());

    // (1) VF contacts + dmin_v/t
    resetCounts<<<vgrid,vblock>>>(C.vfCount, M.nV);
    detectVF<<<vgrid,vblock>>>(M, P.r, rq, B, C);
    CUDA_OK(cudaGetLastError());

    // (2) EE contacts + dmin_e
    resetCounts<<<egrid,eblock>>>(C.eeCount, M.nE);
    detectEE<<<egrid,eblock>>>(M, P.r, rq, B, C);
    CUDA_OK(cudaGetLastError());

    // store X_prev
    CUDA_OK(cudaMemcpy(M.xPrev, M.x, M.nV*sizeof(float3), cudaMemcpyDeviceToDevice));

    // compute bounds
    computeBounds<<<vgrid,vblock>>>(M, B, P);
    CUDA_OK(cudaGetLastError());
    collisionDetectionRequired = false;
  }

  // build inertia target Y (from last committed X, v)
  buildY<<<vgrid,vblock>>>(M, P, B.Y);

  // initial guess truncated (only on first iteration in paper; here always safe)
  applyInitialGuessTruncated<<<vgrid,vblock>>>(M, B.Y, B);

  // iterations
  for (int it=0; it<nIter; ++it){
    zeroFH<<<vgrid,vblock>>>(B.f, B.H, M.nV);

    // inertia
    addInertia<<<vgrid,vblock>>>(M, B.Y, B, P);

    // stretch
    addStretch<<<egrid,eblock>>>(M, B, P);

    // contacts
    applyVFContacts<<<vgrid,vblock>>>(M, C, B, P);
    applyEEContacts<<<egrid,eblock>>>(M, C, B, P);

    // update x
    updatePositions<<<vgrid,vblock>>>(M, B);

    // truncate to bounds & count exceeds
    CUDA_OK(cudaMemset(B.numExceed, 0, sizeof(int)));
    truncateToBounds<<<vgrid,vblock>>>(M, B);

    // host check exceed threshold
    int hEx=0; CUDA_OK(cudaMemcpy(&hEx, B.numExceed, sizeof(int), cudaMemcpyDeviceToHost));
    if (hEx >= (int)std::ceil(P.gamma_e * M.nV)){
      // redo contact detection next outer step
      collisionDetectionRequired = true;
      // refresh X_prev for next bound window
      CUDA_OK(cudaMemcpy(M.xPrev, M.x, M.nV*sizeof(float3), cudaMemcpyDeviceToDevice));
    }
  }

  // update velocities (simple finite difference)
  // v = (x - x_prev_frame)/dt; here assume caller provides previous frame positions if needed.
}

// ---------- Tiny demo: a cloth above a sphere ----------
static HostMesh makeGridCloth(int nx, int ny, float dx, float dy, float z, float massPerVertex){
  HostMesh H{};
  H.X.reserve(nx*ny); H.V.resize(nx*ny, makef3(0,0,0)); H.M.resize(nx*ny, massPerVertex);
  for (int j=0;j<ny;++j){
    for (int i=0;i<nx;++i){
      float x = (i - 0.5f*(nx-1))*dx;
      float y = (j - 0.5f*(ny-1))*dy;
      H.X.push_back(makef3(x,y,z));
    }
  }
  auto vid = [&](int i,int j){ return j*nx+i; };
  for (int j=0;j<ny-1;++j){
    for (int i=0;i<nx-1;++i){
      int v00=vid(i,j), v10=vid(i+1,j), v01=vid(i,j+1), v11=vid(i+1,j+1);
      H.Tris.push_back({v00,v10,v11});
      H.Tris.push_back({v00,v11,v01});
    }
  }
  return H;
}

int main(){
  // Scene: small cloth above origin; a "sphere" collider can be emulated by adding static triangles if desired.
  HostMesh H = makeGridCloth(20, 20, 0.05f, 0.05f, 0.5f, 0.02f);

  // Simple pin (zero mass) at two top corners (optional) – just use very large mass elsewhere
  // Here, keep all dynamic for brevity.

  DeviceMesh M = uploadMesh(H);

  SimParams P{};
  P.dt = 1.f/240.f;
  P.aext = makef3(0, -9.81f, 0);
  P.kc = 1e4f;         // collision stiffness
  P.r  = 0.01f;        // 1 cm contact radius
  P.gamma_p = 0.45f;   // conservative bound relaxation
  P.gamma_e = 0.01f;   // % vertices exceeding -> refresh detection
  P.spring_k = 2e3f;   // edge stretch stiffness
  P.mu = 0.3f; P.eps_v = 1e-2f;

  IterBuffers B = allocIterBuffers(M);
  ContactBuffers C = allocContactBuffers(M);

  // simple loop
  bool needDetect = true;
  const int substeps = 400;          // number of steps
  const int itersPerStep = 15;       // VBD iterations per step

  for (int s=0; s<substeps; ++s){
    step(M, B, C, P, itersPerStep, needDetect);
    // Very crude ground "sphere" repel via OGC can be achieved by meshing the sphere and merging meshes.
    // For now, just print center vertex height:
    if (s % 40 == 0){
      float3 c;
      CUDA_OK(cudaMemcpy(&c, M.x + (M.nV/2), sizeof(float3), cudaMemcpyDeviceToHost));
      printf("step %d, center z=%.4f\n", s, c.z);
    }
  }

  freeContactBuffers(C);
  freeIterBuffers(B);
  freeDeviceMesh(M);
  return 0;
}
