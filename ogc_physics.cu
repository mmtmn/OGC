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

// ---------- OpenGL / Window ----------
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/glu.h>

// ---------- CUDA helpers ----------
#define CUDA_OK(stmt) do { cudaError_t err = (stmt); if (err != cudaSuccess) { \
  fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1);} } while(0)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------- float helpers ----------
static __host__ __device__ inline float3 makef3(float x,float y,float z){ float3 r; r.x=x; r.y=y; r.z=z; return r; }

static __host__ __device__ inline float3 operator+(const float3&a,const float3&b){ return makef3(a.x+b.x,a.y+b.y,a.z+b.z); }
static __host__ __device__ inline float3 operator-(const float3&a,const float3&b){ return makef3(a.x-b.x,a.y-b.y,a.z-b.z); }
static __host__ __device__ inline float3 operator*(const float3&a,float s){ return makef3(a.x*s,a.y*s,a.z*s); }
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

// atomicMin for float via CAS
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
struct Edge { int v0, v1; int t0, t1; int opp0, opp1; };
struct TriEdges { int e01, e12, e20; };

struct CSR { int* offsets; int* indices; };

struct DeviceMesh {
  int nV, nT, nE;

  float3* x;         // positions
  float3* xPrev;     // previous contact-detection anchor (trust region)
  float3* xFramePrev;// previous frame positions (for velocity update)
  float3* v;         // velocities
  float*  m;         // mass per vertex

  uint8_t* isKin;    // kinematic mask per vertex (1=kinematic)

  Tri*  tris;        // triangles
  Edge* edges;       // edges
  TriEdges* triEdges;// tri->edge mapping

  CSR v2n; CSR v2t; CSR v2e;

  float*  edgeRest;      // rest length (for stretch)
  uint8_t* edgeIsSpring; // 1=spring active (cloth), 0=not (static collider edges)
};

// ---------- Contacts ----------
enum FaceType : int { FACE_TRI_INTERIOR = 0, FACE_EDGE = 1, FACE_VERTEX = 2 };
struct VFContact { int tri; int type; int sub; };
struct EEContact { int eOther; int type; int vEndpoint; };

#ifndef MAX_VF_CONTACTS
#define MAX_VF_CONTACTS 64
#endif
#ifndef MAX_EE_CONTACTS
#define MAX_EE_CONTACTS 16
#endif

struct ContactBuffers {
  int* vfCount;      // nV
  VFContact* vfList; // nV * MAX_VF_CONTACTS

  int* eeCount;      // nE
  EEContact* eeList; // nE * MAX_EE_CONTACTS
};

// ---------- Iteration buffers ----------
struct IterBuffers {
  float* dmin_v; float* dmin_t; float* dmin_e; // for bounds
  float* b;                                     // per-vertex trust radius
  float3* Y;                                    // inertia target
  float3* f; float* H;                          // residual & diag
  int* numExceed;                               // counter
};

struct SimParams {
  float dt;
  float3 aext;
  float kc;     // collision stiffness
  float r;      // contact radius
  float gamma_p;
  float gamma_e;
  float spring_k;
  float mu;     // (ext point) friction
  float eps_v;  // (ext point) regularizer
};

// ---------- Closest-point helper ----------
struct ClosestTriResult {
  float3 c;
  float d;
  int type;
  int sub;
  float w0,w1,w2;
};

static __device__ inline ClosestTriResult closestPointTriangle(const float3& p, const float3& a, const float3& b, const float3& c){
  ClosestTriResult R{};
  const float3 ab = b - a, ac = c - a, ap = p - a;
  float d1 = dot(ab, ap);
  float d2 = dot(ac, ap);
  if (d1 <= 0.f && d2 <= 0.f) { R.c=a; R.w0=1;R.w1=0;R.w2=0; R.type=FACE_VERTEX; R.sub=0; R.d=len(p-R.c); return R; }
  const float3 bp = p - b;
  float d3 = dot(ab, bp);
  float d4 = dot(ac, bp);
  if (d3 >= 0.f && d4 <= d3) { R.c=b; R.w0=0;R.w1=1;R.w2=0; R.type=FACE_VERTEX; R.sub=1; R.d=len(p-R.c); return R; }
  float vc = d1*d4 - d3*d2;
  if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f){
    float v = d1 / (d1 - d3);
    R.c = a + ab * v; R.w0=1.f-v; R.w1=v; R.w2=0; R.type=FACE_EDGE; R.sub=0; R.d=len(p-R.c); return R;
  }
  const float3 cp = p - c;
  float d5 = dot(ab, cp);
  float d6 = dot(ac, cp);
  if (d6 >= 0.f && d5 <= d6) { R.c=c; R.w0=0;R.w1=0;R.w2=1; R.type=FACE_VERTEX; R.sub=2; R.d=len(p-R.c); return R; }
  float vb = d5*d2 - d1*d6;
  if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f){
    float w = d2 / (d2 - d6);
    R.c = a + ac * w; R.w0=1.f-w; R.w1=0; R.w2=w; R.type=FACE_EDGE; R.sub=2; R.d=len(p-R.c); return R;
  }
  float va = d3*d6 - d5*d4;
  if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f){
    float w = (d4 - d3)/((d4 - d3) + (d5 - d6));
    R.c = b + (c - b)*w; R.w0=0; R.w1=1.f-w; R.w2=w; R.type=FACE_EDGE; R.sub=1; R.d=len(p-R.c); return R;
  }
  float denom = 1.0f / (va + vb + vc);
  float v = vb * denom, w = vc * denom;
  R.w0 = 1.f - v - w; R.w1=v; R.w2=w;
  R.c = a*R.w0 + b*R.w1 + c*R.w2;
  R.type = FACE_TRI_INTERIOR; R.sub=-1;
  R.d = len(p - R.c);
  return R;
}

static __device__ inline float3 footOnLine(const float3& x1, const float3& x2, const float3& x3){
  float3 d = x2 - x1; float L2 = len2(d);
  if (L2 <= 0.f) return x1;
  float t = dot(x3 - x1, d) / L2;
  return x1 + d * t;
}

static __device__ inline float segSegClosest(const float3& p1, const float3& q1, const float3& p2, const float3& q2,
                                            float& s, float& t, float3& c1, float3& c2)
{
  const float3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
  const float a = dot(d1,d1), e = dot(d2,d2), f = dot(d2,r);
  float EPS=1e-12f;
  if (a <= EPS && e <= EPS) { s=t=0; c1=p1; c2=p2; return len(c1-c2); }
  if (a <= EPS) { s=0; t = f/e; t=fmaxf(0.f,fminf(1.f,t)); }
  else {
    float c = dot(d1,r);
    if (e <= EPS) { t=0; s=fmaxf(0.f, fminf(1.f, -c/a)); }
    else {
      float b = dot(d1,d2), denom = a*e - b*b;
      s = (denom!=0.f)? fmaxf(0.f,fminf(1.f, (b*f - c*e)/denom)) : 0.f;
      t = (b*s + f)/e;
      if (t < 0.f){ t=0; s=fmaxf(0.f, fminf(1.f, -c/a)); }
      else if (t > 1.f){ t=1; s=fmaxf(0.f, fminf(1.f, (b - c)/a)); }
    }
  }
  c1 = p1 + d1*s; c2 = p2 + d2*t;
  return len(c1 - c2);
}

// 2-stage activation derivative g'(d); tau = r/2; kcp = kc * (r^2/4) (C^2 stitch)
static __host__ __device__ inline float activation_dEd_d(const float d, const float r, const float kc){
  if (d >= r) return 0.f;
  const float tau = 0.5f * r;
  const float kcp = kc * (r*r * 0.25f);
  if (d >= tau){
    return -kc * (r - d);          // quadratic stage
  }else{
    float dd = fmaxf(d, 1e-6f);    // barrier stage
    return -(kcp / dd);
  }
}

// Small helpers as device functions (no lambdas)
__device__ inline bool triHasVertex(const Tri& t, int vi){
  return (t.v0==vi || t.v1==vi || t.v2==vi);
}
__device__ inline bool edgesShareVertex(const Edge& a, const Edge& b){
  return (a.v0==b.v0 || a.v0==b.v1 || a.v1==b.v0 || a.v1==b.v1);
}

// Feasible regions (Eqs. 8, 9)
static __device__ inline bool checkVertexFeasibleRegion(const DeviceMesh& M, int vIdx, const float3& xq, float r){
  const float3 xv = M.x[vIdx];
  float3 dv = xq - xv;
  if (len(dv) > r + 1e-6f) return false;
  const int start = M.v2n.offsets[vIdx];
  const int end   = M.v2n.offsets[vIdx+1];
  for (int it=start; it<end; ++it){
    int vnb = M.v2n.indices[it];
    float3 nplane = xv - M.x[vnb];
    if (dot(dv, nplane) < -1e-8f) return false;
  }
  return true;
}

static __device__ inline bool checkEdgeFeasibleRegion(const DeviceMesh& M, int eIdx, const float3& xq, float r){
  const Edge e = M.edges[eIdx];
  const float3 x1 = M.x[e.v0], x2 = M.x[e.v1];
  if (dot(xq - x1, x2 - x1) <= 0.f) return false;
  if (dot(xq - x2, x1 - x2) <= 0.f) return false;

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

// ---------- Detection ----------
__global__ void resetDMin(float* arr, int n, float rq){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) arr[i]=rq; }
__global__ void resetCounts(int* cnt, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) cnt[i]=0; }

__global__ void detectVF(const DeviceMesh M, const float r, const float rq, IterBuffers B, ContactBuffers C)
{
  int v = blockIdx.x*blockDim.x + threadIdx.x; if (v >= M.nV) return;
  float3 xv = M.x[v]; float dmin = rq;

  VFContact localList[MAX_VF_CONTACTS]; int localN=0;

  for (int tIdx=0; tIdx<M.nT; ++tIdx){
    Tri t = M.tris[tIdx];
    if (triHasVertex(t, v)) continue;

    float3 a = M.x[t.v0], b = M.x[t.v1], c = M.x[t.v2];
    ClosestTriResult R = closestPointTriangle(xv, a,b,c);
    dmin = fminf(dmin, R.d);
    atomicMinFloat(&B.dmin_t[tIdx], R.d);

    if (R.d < r){
      int type = R.type, sub = R.sub;
      bool accept=false;
      if (type==FACE_TRI_INTERIOR){
        accept = true;
      } else if (type==FACE_EDGE){
        TriEdges te = M.triEdges[tIdx];
        int ge = (sub==0? te.e01 : (sub==1? te.e12 : te.e20));
        if (ge>=0 && checkEdgeFeasibleRegion(M, ge, xv, r)) accept=true;
      } else { // vertex
        int gv = (sub==0? t.v0 : (sub==1? t.v1 : t.v2));
        if (checkVertexFeasibleRegion(M, gv, xv, r)) accept=true;
      }
      if (accept && localN < MAX_VF_CONTACTS){
        // dedup coarse: avoid same (tri,type,sub)
        bool dup=false;
        for(int i=0;i<localN;++i){ if (localList[i].tri==tIdx && localList[i].type==type && localList[i].sub==sub){ dup=true; break; } }
        if (!dup){ VFContact cc{tIdx,type,sub}; localList[localN++]=cc; }
      }
    }
  }
  B.dmin_v[v] = dmin;

  int base = v*MAX_VF_CONTACTS; C.vfCount[v]=localN;
  for (int i=0;i<localN;++i) C.vfList[base+i] = localList[i];
}

__global__ void detectEE(const DeviceMesh M, const float r, const float rq, IterBuffers B, ContactBuffers C)
{
  int e = blockIdx.x*blockDim.x + threadIdx.x; if (e>=M.nE) return;
  Edge E = M.edges[e];
  float3 p1 = M.x[E.v0], q1 = M.x[E.v1];
  float dmin = rq;

  EEContact localList[MAX_EE_CONTACTS]; int localN=0;

  for (int j=0;j<M.nE;++j){
    if (j==e) continue;
    Edge F = M.edges[j];
    if (edgesShareVertex(E,F)) continue;

    float3 p2 = M.x[F.v0], q2 = M.x[F.v1];
    float s,t; float3 c1,c2;
    float d = segSegClosest(p1,q1,p2,q2,s,t,c1,c2);
    dmin = fminf(dmin, d);
    if (d < r){
      int type=0, vend=-1;
      if (s < 1e-3f) { type=1; vend=E.v0; }
      else if (s > 1.f-1e-3f) { type=1; vend=E.v1; }
      if (type==1){
        if (!checkVertexFeasibleRegion(M, vend, c1, r)) continue;
      }
      if (localN < MAX_EE_CONTACTS){
        EEContact ce{j,type,vend};
        localList[localN++] = ce;
      }
    }
  }
  B.dmin_e[e] = dmin;
  int base = e*MAX_EE_CONTACTS; C.eeCount[e]=localN;
  for (int i=0;i<localN;++i) C.eeList[base+i] = localList[i];
}

// ---------- Bounds ----------
__global__ void computeBounds(const DeviceMesh M, IterBuffers B, const SimParams P){
  int v = blockIdx.x*blockDim.x + threadIdx.x; if (v>=M.nV) return;

  float dmin_v = B.dmin_v[v];

  float dEmin = 1e30f;
  int sE = M.v2e.offsets[v], eE = M.v2e.offsets[v+1];
  for (int it=sE; it<eE; ++it){ int eIdx = M.v2e.indices[it]; dEmin = fminf(dEmin, B.dmin_e[eIdx]); }

  float dTmin = 1e30f;
  int sT = M.v2t.offsets[v], eT = M.v2t.offsets[v+1];
  for (int it=sT; it<eT; ++it){ int t = M.v2t.indices[it]; dTmin = fminf(dTmin, B.dmin_t[t]); }

  float dmin_all = fminf(dmin_v, fminf(dEmin, dTmin));
  dmin_all = fmaxf(dmin_all, 0.0f);
  B.b[v] = P.gamma_p * dmin_all;
}

// ---------- Inertia / initial guess ----------
__global__ void buildY(const DeviceMesh M, const SimParams P, float3* Y){
  int v = blockIdx.x*blockDim.x + threadIdx.x; if (v>=M.nV) return;
  if (M.isKin[v]){ Y[v] = M.x[v]; return; } // kinematic stays
  Y[v] = M.x[v] + M.v[v]*P.dt + P.aext*(P.dt*P.dt);
}

__global__ void applyInitialGuessTruncated(DeviceMesh M, const float3* Y, IterBuffers B){
  int v = blockIdx.x*blockDim.x + threadIdx.x; if (v>=M.nV) return;
  float3 xinit = Y[v];
  float3 xprev = M.xPrev[v];
  float3 d = xinit - xprev;
  float L = len(d);
  float bv = B.b[v];
  float3 xstar = (L <= bv || L <= 1e-12f) ? xinit : (xprev + d*(bv/L));
  ((float3*)M.x)[v] = xstar;
}

// ---------- Solver accum ----------
__global__ void zeroFH(float3* f, float* H, int nV){
  int v = blockIdx.x*blockDim.x + threadIdx.x; if (v<nV){ f[v]=makef3(0,0,0); H[v]=0.f; }
}

__global__ void addInertia(const DeviceMesh M, const float3* Y, IterBuffers B, const SimParams P){
  int v = blockIdx.x*blockDim.x + threadIdx.x; if (v>=M.nV) return;
  if (M.isKin[v]) return;
  float invh2 = 1.f / (P.dt*P.dt);
  float w = M.m[v] * invh2;
  float3 r = M.x[v] - Y[v];
  atomicAddFloat3(&B.f[v], makef3(-w*r.x, -w*r.y, -w*r.z));
  atomicAdd(&B.H[v], w);
}

__global__ void addStretch(const DeviceMesh M, IterBuffers B, const SimParams P){
  int e = blockIdx.x*blockDim.x + threadIdx.x; if (e>=M.nE) return;
  if (!M.edgeIsSpring[e]) return;
  Edge E = M.edges[e]; int i=E.v0, j=E.v1;
  float3 xi = M.x[i], xj = M.x[j]; float3 d = xi - xj; float L = len(d);
  float L0 = M.edgeRest[e];
  float3 dir = (L>1e-9f)? (d/L) : makef3(0,0,0);
  float k = P.spring_k;
  float coeff = k * (1.f - (L0 / fmaxf(L, 1e-9f)));
  float3 fi = dir * coeff, fj = fi * (-1.f);

  if (!M.isKin[i]){ atomicAddFloat3(&B.f[i], fi); atomicAdd(&B.H[i], k); }
  if (!M.isKin[j]){ atomicAddFloat3(&B.f[j], fj); atomicAdd(&B.H[j], k); }
}

// VF contacts
__global__ void applyVFContacts(const DeviceMesh M, const ContactBuffers C, IterBuffers B, const SimParams P){
  int v = blockIdx.x*blockDim.x + threadIdx.x; if (v>=M.nV) return;
  int n = C.vfCount[v]; int base = v*MAX_VF_CONTACTS; float3 xv=M.x[v];

  for (int k=0;k<n;++k){
    VFContact c = C.vfList[base+k]; Tri t = M.tris[c.tri];
    float3 a=M.x[t.v0], b=M.x[t.v1], cc=M.x[t.v2];

    float3 Cp; float d; float3 nrm; float w0=0,w1=0,w2=0;
    if (c.type == FACE_TRI_INTERIOR){
      ClosestTriResult R = closestPointTriangle(xv,a,b,cc);
      Cp=R.c; d=R.d; nrm=normalize(xv - Cp); w0=R.w0; w1=R.w1; w2=R.w2;
    } else if (c.type == FACE_EDGE){
      int ge = (c.sub==0? M.triEdges[c.tri].e01 : (c.sub==1? M.triEdges[c.tri].e12 : M.triEdges[c.tri].e20));
      Edge E = M.edges[ge];
      float s,t; float3 c1,c2; d=segSegClosest(xv,xv,M.x[E.v0],M.x[E.v1],s,t,c1,c2);
      Cp = c2; nrm=normalize(xv - Cp);
    } else {
      int gv = (c.sub==0? t.v0 : (c.sub==1? t.v1 : t.v2));
      Cp = M.x[gv]; d=len(xv - Cp); nrm=normalize(xv - Cp);
    }

    float dEd = activation_dEd_d(d, P.r, P.kc);
    float Fn = -dEd;
    float3 F = nrm * Fn; // on v

    if (!M.isKin[v]){ atomicAddFloat3(&B.f[v], F); atomicAdd(&B.H[v], P.kc); }

    // Reaction:
    float3 Fr = F * (-1.f);
    if (c.type == FACE_TRI_INTERIOR){
      if (!M.isKin[t.v0]){ atomicAddFloat3(&B.f[t.v0], Fr*w0); atomicAdd(&B.H[t.v0], P.kc*0.3333f); }
      if (!M.isKin[t.v1]){ atomicAddFloat3(&B.f[t.v1], Fr*w1); atomicAdd(&B.H[t.v1], P.kc*0.3333f); }
      if (!M.isKin[t.v2]){ atomicAddFloat3(&B.f[t.v2], Fr*w2); atomicAdd(&B.H[t.v2], P.kc*0.3333f); }
    } else if (c.type == FACE_EDGE){
      int ge = (c.sub==0? M.triEdges[c.tri].e01 : (c.sub==1? M.triEdges[c.tri].e12 : M.triEdges[c.tri].e20));
      Edge E = M.edges[ge]; float3 evec=M.x[E.v1]-M.x[E.v0]; float L2=len2(evec);
      float tpar = (L2>0.f)? dot(Cp - M.x[E.v0], evec)/L2 : 0.f; tpar=fmaxf(0.f,fminf(1.f,tpar));
      if (!M.isKin[E.v0]){ atomicAddFloat3(&B.f[E.v0], Fr*(1.f - tpar)); atomicAdd(&B.H[E.v0], P.kc*0.5f); }
      if (!M.isKin[E.v1]){ atomicAddFloat3(&B.f[E.v1], Fr*(tpar));       atomicAdd(&B.H[E.v1], P.kc*0.5f); }
    } else {
      int gv = (c.sub==0? t.v0 : (c.sub==1? t.v1 : t.v2));
      if (!M.isKin[gv]){ atomicAddFloat3(&B.f[gv], Fr); atomicAdd(&B.H[gv], P.kc); }
    }
    // (Friction extension would go here.)
  }
}

// EE contacts
__global__ void applyEEContacts(const DeviceMesh M, const ContactBuffers C, IterBuffers B, const SimParams P){
  int e = blockIdx.x*blockDim.x + threadIdx.x; if (e>=M.nE) return;
  int n = C.eeCount[e]; int base = e*MAX_EE_CONTACTS; Edge E = M.edges[e];
  float3 p1=M.x[E.v0], q1=M.x[E.v1];

  for (int k=0;k<n;++k){
    EEContact ce = C.eeList[base+k];
    Edge F = M.edges[ce.eOther]; float3 p2=M.x[F.v0], q2=M.x[F.v1];
    float s,t; float3 c1,c2; float d=segSegClosest(p1,q1,p2,q2,s,t,c1,c2);
    float3 nrm = normalize(c1 - c2);
    float dEd = activation_dEd_d(d, P.r, P.kc); float Fn = -dEd;
    float3 F12 = nrm * Fn;
    float wi0=1.f - s, wi1=s; float wj0=1.f - t, wj1=t;

    if (!M.isKin[E.v0]){ atomicAddFloat3(&B.f[E.v0], F12*wi0); atomicAdd(&B.H[E.v0], P.kc*0.5f); }
    if (!M.isKin[E.v1]){ atomicAddFloat3(&B.f[E.v1], F12*wi1); atomicAdd(&B.H[E.v1], P.kc*0.5f); }

    float3 Fr = F12 * (-1.f);
    if (!M.isKin[F.v0]){ atomicAddFloat3(&B.f[F.v0], Fr*wj0); atomicAdd(&B.H[F.v0], P.kc*0.5f); }
    if (!M.isKin[F.v1]){ atomicAddFloat3(&B.f[F.v1], Fr*wj1); atomicAdd(&B.H[F.v1], P.kc*0.5f); }
  }
}

__global__ void updatePositions(DeviceMesh M, IterBuffers B){
  int v = blockIdx.x*blockDim.x + threadIdx.x; if (v>=M.nV) return;
  if (M.isKin[v]) return;
  float H = fmaxf(B.H[v], 1e-9f);
  float3 dx = B.f[v] / H;
  ((float3*)M.x)[v] = M.x[v] + dx;
}

__global__ void truncateToBounds(const DeviceMesh M, IterBuffers B){
  int v = blockIdx.x*blockDim.x + threadIdx.x; if (v>=M.nV) return;
  if (M.isKin[v]) return;
  float3 xprev = M.xPrev[v];
  float3 d = M.x[v] - xprev;
  float L = len(d);
  float b = B.b[v];
  if (L > b && L > 1e-12f){
    ((float3*)M.x)[v] = xprev + d * (b / L);
    atomicAdd(B.numExceed, 1);
  }
}

__global__ void updateVelocities(DeviceMesh M, const SimParams P){
  int v = blockIdx.x*blockDim.x + threadIdx.x; if (v>=M.nV) return;
  if (M.isKin[v]){ M.v[v]=makef3(0,0,0); return; }
  float3 vel = (M.x[v] - M.xFramePrev[v]) / P.dt;
  M.v[v] = vel;
}

__global__ void storeFramePrev(DeviceMesh M){
  int v = blockIdx.x*blockDim.x + threadIdx.x; if (v>=M.nV) return;
  ((float3*)M.xFramePrev)[v] = M.x[v];
}

// ---------- Host mesh builder ----------
struct HostMesh {
  std::vector<float3> X;
  std::vector<float3> V;
  std::vector<float>  M;
  std::vector<uint8_t> isKin;
  std::vector<Tri>    Tris;
};

static inline uint64_t keyEdge(int a, int b){ uint64_t A=(uint32_t)std::min(a,b); uint64_t B=(uint32_t)std::max(a,b); return (A<<32)|B; }

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

  for (int t=0;t<nT;++t){
    Tri T = H.Tris[t];
    int e01=addEdge(T.v0,T.v1,t,T.v2);
    int e12=addEdge(T.v1,T.v2,t,T.v0);
    int e20=addEdge(T.v2,T.v0,t,T.v1);
    triEdges[t] = TriEdges{e01,e12,e20};
  }

  std::vector<std::vector<int>> VV(nV), VT(nV), VE(nV);
  for (int t=0;t<nT;++t){
    Tri T = H.Tris[t];
    VT[T.v0].push_back(t); VT[T.v1].push_back(t); VT[T.v2].push_back(t);
    VV[T.v0].push_back(T.v1); VV[T.v0].push_back(T.v2);
    VV[T.v1].push_back(T.v0); VV[T.v1].push_back(T.v2);
    VV[T.v2].push_back(T.v0); VV[T.v2].push_back(T.v1);
  }
  for (int e=0;e<(int)edges.size();++e){ VE[edges[e].v0].push_back(e); VE[edges[e].v1].push_back(e); }

  auto uniqSort = [](std::vector<int>& v){ std::sort(v.begin(),v.end()); v.erase(std::unique(v.begin(),v.end()), v.end()); };

  v2n_offsets.resize(nV+1); v2t_offsets.resize(nV+1); v2e_offsets.resize(nV+1);
  int accN=0,accT=0,accE=0;
  for (int v=0; v<nV; ++v){
    uniqSort(VV[v]); uniqSort(VT[v]); uniqSort(VE[v]);
    v2n_offsets[v]=accN; accN+=(int)VV[v].size();
    v2t_offsets[v]=accT; accT+=(int)VT[v].size();
    v2e_offsets[v]=accE; accE+=(int)VE[v].size();
  }
  v2n_offsets[nV]=accN; v2t_offsets[nV]=accT; v2e_offsets[nV]=accE;

  v2n_indices.resize(accN); v2t_indices.resize(accT); v2e_indices.resize(accE);
  for (int v=0; v<nV; ++v){
    int s=v2n_offsets[v];
    for(int i=0;i<(int)VV[v].size();++i) v2n_indices[s+i]=VV[v][i];
    s=v2t_offsets[v];
    for(int i=0;i<(int)VT[v].size();++i) v2t_indices[s+i]=VT[v][i];
    s=v2e_offsets[v];
    for(int i=0;i<(int)VE[v].size();++i) v2e_indices[s+i]=VE[v][i];
  }
}

// Upload host mesh to device (computes edges, triEdges, edge rest, edgeIsSpring)
static DeviceMesh uploadMesh(const HostMesh& H){
  DeviceMesh M{};
  M.nV=(int)H.X.size(); M.nT=(int)H.Tris.size();

  std::vector<Edge> edges;
  std::vector<TriEdges> triEdges;
  std::vector<int> v2n_off,v2n_idx, v2t_off,v2t_idx, v2e_off,v2e_idx;
  buildEdgesAndAdj(H, edges, triEdges, v2n_off,v2n_idx, v2t_off,v2t_idx, v2e_off,v2e_idx);
  M.nE=(int)edges.size();

  // edge rest length and spring flags
  std::vector<float> L0(edges.size());
  std::vector<uint8_t> edgeSpring(edges.size(), 1);
  for (int i=0;i<(int)edges.size();++i){
    int a=edges[i].v0, b=edges[i].v1;
    float3 xa=H.X[a], xb=H.X[b];
    L0[i] = sqrtf((xa.x-xb.x)*(xa.x-xb.x)+(xa.y-xb.y)*(xa.y-xb.y)+(xa.z-xb.z)*(xa.z-xb.z));
    // disable springs for purely kinematic edges (static collider mesh)
    if (H.isKin[a] && H.isKin[b]) edgeSpring[i]=0;
  }

  // allocate device buffers
  CUDA_OK(cudaMalloc(&M.x,      M.nV*sizeof(float3)));
  CUDA_OK(cudaMalloc(&M.xPrev,  M.nV*sizeof(float3)));
  CUDA_OK(cudaMalloc(&M.xFramePrev, M.nV*sizeof(float3)));
  CUDA_OK(cudaMalloc(&M.v,      M.nV*sizeof(float3)));
  CUDA_OK(cudaMalloc(&M.m,      M.nV*sizeof(float)));
  CUDA_OK(cudaMalloc(&M.isKin,  M.nV*sizeof(uint8_t)));

  CUDA_OK(cudaMalloc(&M.tris,    M.nT*sizeof(Tri)));
  CUDA_OK(cudaMalloc(&M.edges,   M.nE*sizeof(Edge)));
  CUDA_OK(cudaMalloc(&M.triEdges,M.nT*sizeof(TriEdges)));

  CUDA_OK(cudaMalloc(&M.v2n.offsets, (M.nV+1)*sizeof(int)));
  CUDA_OK(cudaMalloc(&M.v2n.indices, v2n_idx.size()*sizeof(int)));
  CUDA_OK(cudaMalloc(&M.v2t.offsets, (M.nV+1)*sizeof(int)));
  CUDA_OK(cudaMalloc(&M.v2t.indices, v2t_idx.size()*sizeof(int)));
  CUDA_OK(cudaMalloc(&M.v2e.offsets, (M.nV+1)*sizeof(int)));
  CUDA_OK(cudaMalloc(&M.v2e.indices, v2e_idx.size()*sizeof(int)));

  CUDA_OK(cudaMalloc(&M.edgeRest,      M.nE*sizeof(float)));
  CUDA_OK(cudaMalloc(&M.edgeIsSpring,  M.nE*sizeof(uint8_t)));

  // copy
  CUDA_OK(cudaMemcpy(M.x,     H.X.data(),   M.nV*sizeof(float3), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.xPrev, H.X.data(),   M.nV*sizeof(float3), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.xFramePrev, H.X.data(), M.nV*sizeof(float3), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.v,     H.V.data(),   M.nV*sizeof(float3), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.m,     H.M.data(),   M.nV*sizeof(float),  cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.isKin, H.isKin.data(), M.nV*sizeof(uint8_t), cudaMemcpyHostToDevice));

  CUDA_OK(cudaMemcpy(M.tris,    H.Tris.data(), M.nT*sizeof(Tri), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.edges,   edges.data(),  M.nE*sizeof(Edge), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.triEdges,triEdges.data(), M.nT*sizeof(TriEdges), cudaMemcpyHostToDevice));

  CUDA_OK(cudaMemcpy(M.v2n.offsets, v2n_off.data(), (M.nV+1)*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.v2n.indices, v2n_idx.data(), v2n_idx.size()*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.v2t.offsets, v2t_off.data(), (M.nV+1)*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.v2t.indices, v2t_idx.data(), v2t_idx.size()*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.v2e.offsets, v2e_off.data(), (M.nV+1)*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.v2e.indices, v2e_idx.data(), v2e_idx.size()*sizeof(int), cudaMemcpyHostToDevice));

  CUDA_OK(cudaMemcpy(M.edgeRest,     L0.data(),        M.nE*sizeof(float),   cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(M.edgeIsSpring, edgeSpring.data(),M.nE*sizeof(uint8_t), cudaMemcpyHostToDevice));

  return M;
}

static void freeDeviceMesh(DeviceMesh& M){
  cudaFree(M.x); cudaFree(M.xPrev); cudaFree(M.xFramePrev); cudaFree(M.v); cudaFree(M.m); cudaFree(M.isKin);
  cudaFree(M.tris); cudaFree(M.edges); cudaFree(M.triEdges);
  cudaFree(M.v2n.offsets); cudaFree(M.v2n.indices);
  cudaFree(M.v2t.offsets); cudaFree(M.v2t.indices);
  cudaFree(M.v2e.offsets); cudaFree(M.v2e.indices);
  cudaFree(M.edgeRest); cudaFree(M.edgeIsSpring);
  M = DeviceMesh{};
}

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

// ---------- Simulation step ----------
static void step(DeviceMesh& M, IterBuffers& B, ContactBuffers& C, const SimParams& P, int nIter,
                 bool& collisionDetectionRequired)
{
  dim3 vgrid((M.nV+127)/128), vblock(128);
  dim3 tgrid((M.nT+127)/128), tblock(128);
  dim3 egrid((M.nE+127)/128), eblock(128);

  if (collisionDetectionRequired){
    float rq = P.r + 2.0f * len(P.aext) * P.dt * P.dt + 2.0f * P.dt; // heuristic
    resetDMin<<<tgrid,tblock>>>(B.dmin_t, M.nT, rq);
    CUDA_OK(cudaGetLastError());

    resetCounts<<<vgrid,vblock>>>(C.vfCount, M.nV);
    detectVF<<<vgrid,vblock>>>(M, P.r, rq, B, C);
    CUDA_OK(cudaGetLastError());

    resetCounts<<<egrid,eblock>>>(C.eeCount, M.nE);
    detectEE<<<egrid,eblock>>>(M, P.r, rq, B, C);
    CUDA_OK(cudaGetLastError());

    CUDA_OK(cudaMemcpy(M.xPrev, M.x, M.nV*sizeof(float3), cudaMemcpyDeviceToDevice));
    computeBounds<<<vgrid,vblock>>>(M, B, P);
    CUDA_OK(cudaGetLastError());

    collisionDetectionRequired=false;
  }

  buildY<<<vgrid,vblock>>>(M, P, B.Y);
  applyInitialGuessTruncated<<<vgrid,vblock>>>(M, B.Y, B);

  for (int it=0; it<nIter; ++it){
    zeroFH<<<vgrid,vblock>>>(B.f, B.H, M.nV);

    addInertia<<<vgrid,vblock>>>(M, B.Y, B, P);
    addStretch<<<egrid,eblock>>>(M, B, P);
    applyVFContacts<<<vgrid,vblock>>>(M, C, B, P);
    applyEEContacts<<<egrid,eblock>>>(M, C, B, P);

    updatePositions<<<vgrid,vblock>>>(M, B);

    CUDA_OK(cudaMemset(B.numExceed, 0, sizeof(int)));
    truncateToBounds<<<vgrid,vblock>>>(M, B);

    int hEx=0; CUDA_OK(cudaMemcpy(&hEx, B.numExceed, sizeof(int), cudaMemcpyDeviceToHost));
    if (hEx >= (int)std::ceil(P.gamma_e * M.nV)){
      collisionDetectionRequired = true;
      CUDA_OK(cudaMemcpy(M.xPrev, M.x, M.nV*sizeof(float3), cudaMemcpyDeviceToDevice));
    }
  }

  // velocity update from last frame positions
  updateVelocities<<<vgrid,vblock>>>(M, P);
  storeFramePrev<<<vgrid,vblock>>>(M);
}

// ---------- World building helpers ----------
static HostMesh makeClothGrid(int nx, int ny, float dx, float dy, float3 center, float massPerVertex, bool kinematic=false){
  HostMesh H{};
  H.X.reserve(nx*ny); H.V.assign(nx*ny, makef3(0,0,0)); H.M.assign(nx*ny, massPerVertex); H.isKin.assign(nx*ny, kinematic?1:0);
  for (int j=0;j<ny;++j){
    for (int i=0;i<nx;++i){
      float x = (i - 0.5f*(nx-1))*dx + center.x;
      float y = (j - 0.5f*(ny-1))*dy + center.y;
      float z = center.z;
      H.X.push_back(makef3(x,y,z));
    }
  }
  auto vid=[&](int i,int j){ return j*nx+i; };
  for (int j=0;j<ny-1;++j){
    for (int i=0;i<nx-1;++i){
      int v00=vid(i,j), v10=vid(i+1,j), v01=vid(i,j+1), v11=vid(i+1,j+1);
      H.Tris.push_back({v00,v10,v11});
      H.Tris.push_back({v00,v11,v01});
    }
  }
  return H;
}

static HostMesh makeUVSphere(int slices, int stacks, float radius, float3 center, bool kinematic=true){
  HostMesh H{};
  for (int j=0;j<=stacks;++j){
    float v = (float)j / (float)stacks;
    float phi = v * M_PI; // 0..pi
    for (int i=0;i<=slices;++i){
      float u = (float)i / (float)slices;
      float theta = u * 2.f * M_PI; // 0..2pi
      float xs = sinf(phi) * cosf(theta);
      float ys = cosf(phi);
      float zs = sinf(phi) * sinf(theta);
      H.X.push_back( makef3(center.x + radius*xs, center.y + radius*ys, center.z + radius*zs) );
      H.V.push_back( makef3(0,0,0) );
      H.M.push_back( kinematic? 0.f : 1.f );
      H.isKin.push_back( kinematic? 1:0 );
    }
  }
  int stride = slices+1;
  for (int j=0;j<stacks;++j){
    for (int i=0;i<slices;++i){
      int i0 = j*stride + i;
      int i1 = i0 + 1;
      int i2 = i0 + stride;
      int i3 = i2 + 1;
      H.Tris.push_back({i0,i2,i1});
      H.Tris.push_back({i1,i2,i3});
    }
  }
  return H;
}

// Merge A += B (adjust indices)
static void appendMesh(HostMesh& A, const HostMesh& B){
  int off = (int)A.X.size();
  A.X.insert(A.X.end(), B.X.begin(), B.X.end());
  A.V.insert(A.V.end(), B.V.begin(), B.V.end());
  A.M.insert(A.M.end(), B.M.begin(), B.M.end());
  A.isKin.insert(A.isKin.end(), B.isKin.begin(), B.isKin.end());
  for (auto t : B.Tris){ A.Tris.push_back({t.v0+off, t.v1+off, t.v2+off}); }
}

// ---------- OpenGL camera ----------
struct Camera {
  float3 pos{0.f, 1.0f, 3.0f};
  float yaw{-90.f};  // degrees (0 along +X)
  float pitch{ -10.f};
  float moveSpeed{2.5f};
  float mouseSens{0.1f};
  bool firstMouse=true;
  double lastX=0,lastY=0;

  float3 forward() const {
    float cy=cosf(yaw*M_PI/180.f), sy=sinf(yaw*M_PI/180.f);
    float cp=cosf(pitch*M_PI/180.f), sp=sinf(pitch*M_PI/180.f);
    float3 f = makef3(cy*cp, sp, sy*cp);
    return normalize(f);
  }
  float3 right() const {
    return normalize(cross(forward(), makef3(0,1,0)));
  }
  float3 up() const {
    return normalize(cross(right(), forward()*(-1.f)));
  }
};

static Camera gCam;
static bool gWire=false;
static bool gPaused=false;

// input debounce
struct KeyLatch { bool prev=false; };
static bool press(GLFWwindow* W, int key, KeyLatch& latch){
  int state = glfwGetKey(W, key);
  bool now = (state==GLFW_PRESS);
  bool fired = (now && !latch.prev);
  latch.prev = now;
  return fired;
}

static void mouseCB(GLFWwindow* win, double xpos, double ypos){
    if (gCam.firstMouse){ gCam.lastX=xpos; gCam.lastY=ypos; gCam.firstMouse=false; }
    float dx = float(gCam.lastX - xpos);   // flip yaw
    float dy = float(ypos - gCam.lastY);   // flip pitch

    gCam.lastX = xpos;
    gCam.lastY = ypos;

    dx *= gCam.mouseSens;
    dy *= gCam.mouseSens;

    gCam.yaw   += dx;
    gCam.pitch += dy;

    if (gCam.pitch > 89.f)  gCam.pitch = 89.f;
    if (gCam.pitch < -89.f) gCam.pitch = -89.f;
}

static void applyCameraGL(){
  glMatrixMode(GL_PROJECTION); glLoadIdentity();
  gluPerspective(60.0, 16.0/9.0, 0.05, 200.0);
  glMatrixMode(GL_MODELVIEW); glLoadIdentity();
  float3 f = gCam.forward();
  float3 u = gCam.up();
  float3 c = gCam.pos + f;
  gluLookAt(gCam.pos.x, gCam.pos.y, gCam.pos.z,
            c.x, c.y, c.z,
            u.x, u.y, u.z);
}

// ---------- Renderer (immediate mode for simplicity) ----------
static void drawMeshImmediate(const std::vector<float3>& X, const std::vector<Tri>& T, const std::vector<uint8_t>& isKin){
  glDisable(GL_LIGHTING);
  if (gWire){
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glColor3f(0.9f, 0.9f, 0.9f);
  } else {
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  }

  glBegin(GL_TRIANGLES);
  for (const auto& tri : T){
    // Color by kinematic/dynamic
    bool anyKin = isKin[tri.v0] || isKin[tri.v1] || isKin[tri.v2];
    if (anyKin) glColor3f(0.5f,0.5f,0.5f); else glColor3f(0.2f,0.7f,1.0f);

    const float3& a = X[tri.v0];
    const float3& b = X[tri.v1];
    const float3& c = X[tri.v2];
    glVertex3f(a.x,a.y,a.z);
    glVertex3f(b.x,b.y,b.z);
    glVertex3f(c.x,c.y,c.z);
  }
  glEnd();
}

// ---------- Main ----------
int main(){
  // ---- Init window ----
  if (!glfwInit()){ fprintf(stderr,"glfwInit failed\n"); return 1; }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);

  GLFWwindow* win = glfwCreateWindow(1280, 720, "OGC Lab (CUDA + OpenGL)", nullptr, nullptr);
  if (!win){ fprintf(stderr,"glfwCreateWindow failed\n"); glfwTerminate(); return 1; }
  glfwMakeContextCurrent(win);
  glewExperimental = GL_TRUE;
  if (glewInit()!=GLEW_OK){ fprintf(stderr,"glewInit failed\n"); return 1; }
  glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glfwSetCursorPosCallback(win, mouseCB);
  glEnable(GL_DEPTH_TEST);
  glClearColor(0.06f,0.06f,0.08f,1.f);

  // ---- World (host) ----
  HostMesh Hworld{};
  // initial: one cloth above origin
  appendMesh(Hworld, makeClothGrid(20,20, 0.05f,0.05f, makef3(0,0,1.2f), 0.02f, false));

  // ---- Upload to device ----
  DeviceMesh M = uploadMesh(Hworld);
  IterBuffers B = allocIterBuffers(M);
  ContactBuffers C = allocContactBuffers(M);

  SimParams P{};
  P.dt = 1.f/240.f;
  P.aext = makef3(0,-9.81f,0);
  P.kc = 1e4f;
  P.r  = 0.01f;
  P.gamma_p=0.45f; P.gamma_e=0.01f;
  P.spring_k=2e3f;
  P.mu=0.3f; P.eps_v=1e-2f;

  bool needDetect=true;

  // host buffer for rendering positions
  std::vector<float3> hX(Hworld.X.size());

  // timing
  double lastTime = glfwGetTime();
  double simAcc = 0.0;
  const double fixedDt = P.dt;

  // input latches
  KeyLatch latchSpawnCloth, latchSpawnSphere, latchPause, latchReset, latchWire;

  while (!glfwWindowShouldClose(win)){
    glfwPollEvents();

    // --- camera movement ---
    float3 fwd = gCam.forward(), right = gCam.right();
    double now = glfwGetTime();
    float frameDt = float(now - lastTime);
    lastTime = now;

    float move = gCam.moveSpeed * frameDt;
    if (glfwGetKey(win, GLFW_KEY_W)==GLFW_PRESS) gCam.pos = gCam.pos + fwd*move;
    if (glfwGetKey(win, GLFW_KEY_S)==GLFW_PRESS) gCam.pos = gCam.pos - fwd*move;
    if (glfwGetKey(win, GLFW_KEY_A)==GLFW_PRESS) gCam.pos = gCam.pos + right*move;
    if (glfwGetKey(win, GLFW_KEY_D)==GLFW_PRESS) gCam.pos = gCam.pos - right*move;

    if (glfwGetKey(win, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS) gCam.pos.y -= move;
    if (glfwGetKey(win, GLFW_KEY_SPACE)==GLFW_PRESS && press(win, GLFW_KEY_SPACE, latchPause)) gPaused = !gPaused;

    if (press(win, GLFW_KEY_1, latchSpawnCloth)){
      // download current device positions to host, rebuild world, reupload with new cloth
      CUDA_OK(cudaMemcpy(hX.data(), M.x, M.nV*sizeof(float3), cudaMemcpyDeviceToHost));
      Hworld.X = hX; // keep current state as base
      // ensure sizes match
      Hworld.V.resize(M.nV, makef3(0,0,0));
      Hworld.M.resize(M.nV);
      Hworld.isKin.resize(M.nV);
      // spawn cloth in front
      float3 spawnPos = gCam.pos + fwd*1.0f + makef3(0,0,0.5f);
      HostMesh add = makeClothGrid(20,20, 0.05f,0.05f, spawnPos, 0.02f, false);
      appendMesh(Hworld, add);

      freeContactBuffers(C); freeIterBuffers(B); freeDeviceMesh(M);
      M = uploadMesh(Hworld);
      B = allocIterBuffers(M); C = allocContactBuffers(M);
      needDetect = true;
      hX.resize(M.nV);
    }

    if (press(win, GLFW_KEY_2, latchSpawnSphere)){
      CUDA_OK(cudaMemcpy(hX.data(), M.x, M.nV*sizeof(float3), cudaMemcpyDeviceToHost));
      Hworld.X = hX; Hworld.V.resize(M.nV, makef3(0,0,0));
      Hworld.M.resize(M.nV); Hworld.isKin.resize(M.nV);
      float3 spawnPos = gCam.pos + fwd*1.5f;
      HostMesh sph = makeUVSphere(24, 16, 0.25f, spawnPos, true);
      appendMesh(Hworld, sph);

      freeContactBuffers(C); freeIterBuffers(B); freeDeviceMesh(M);
      M = uploadMesh(Hworld);
      B = allocIterBuffers(M); C = allocContactBuffers(M);
      needDetect = true;
      hX.resize(M.nV);
    }

    if (press(win, GLFW_KEY_R, latchReset)){
      // rebuild a fresh single cloth above origin
      freeContactBuffers(C); freeIterBuffers(B); freeDeviceMesh(M);
      Hworld = HostMesh{};
      appendMesh(Hworld, makeClothGrid(20,20, 0.05f,0.05f, makef3(0,0,1.2f), 0.02f, false));
      M = uploadMesh(Hworld);
      B = allocIterBuffers(M); C = allocContactBuffers(M);
      needDetect = true;
      hX.resize(M.nV);
    }

    if (press(win, GLFW_KEY_F, latchWire)){ gWire = !gWire; }

    // --- physics ---
    if (!gPaused){
      simAcc += frameDt;
      // catch up with fixed dt; clamp to avoid spiral of death
      int maxSub=8; int steps=0;
      while (simAcc >= fixedDt && steps<maxSub){
        step(M, B, C, P, /*iterations*/ 12, needDetect);
        simAcc -= fixedDt; steps++;
      }
    }

    // --- render ---
    CUDA_OK(cudaMemcpy(hX.data(), M.x, M.nV*sizeof(float3), cudaMemcpyDeviceToHost));

    glViewport(0,0,1280,720);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    applyCameraGL();

    // simple grid floor
    glDisable(GL_LIGHTING);
    glColor3f(0.2f,0.2f,0.22f);
    glBegin(GL_LINES);
    for (int i=-10;i<=10;++i){
      glVertex3f((float)i*0.5f, 0, -5.f); glVertex3f((float)i*0.5f, 0, 5.f);
      glVertex3f(-5.f, 0, (float)i*0.5f); glVertex3f(5.f, 0, (float)i*0.5f);
    }
    glEnd();

    drawMeshImmediate(hX, Hworld.Tris, Hworld.isKin);

    // HUD-ish text (skip bitmap fonts to keep dependencies minimal)
    // (Use terminal prints if needed.)

    glfwSwapBuffers(win);
  }

  freeContactBuffers(C); freeIterBuffers(B); freeDeviceMesh(M);
  glfwDestroyWindow(win);
  glfwTerminate();
  return 0;
}
