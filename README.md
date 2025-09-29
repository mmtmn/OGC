The research paper can be find here: https://graphics.cs.utah.edu/research/projects/ogc/Offset_Geometric_Contact-SIGGRAPH2025.pdf

Offset Geometric Contact
ANKA HE CHEN, University of Utah, NVIDIA, USA
JERRY HSU, University of Utah, USA
ZIHENG LIU, University of Utah, USA
MILES MACKLIN, NVIDIA, New Zealand
YIN YANG, University of Utah, USA
CEM YUKSEL, University of Utah, USA

Fig. 1. Example simulation results using our penetration-free contact handling method. Our method is robust in the presence of challenging
contact scenarios, and can be easily integrated with existing solvers such as Vertex Block Descent [Chen et al. 2024b], as shown here.
We present a novel contact model, termed Offset Geometric Contact (OGC),
for guaranteed penetration-free simulation of codimensional objects with
minimal computational overhead. Our method is based on constructing
a volumetric shape by offsetting each face along its normal direction, en-
suring orthogonal contact forces, thus allows large contact radius without
artifacts. We compute vertex-specific displacement bounds to guarantee
penetration-free simulation, which improves convergence and avoids the
need for expensive continuous collision detection. Our method relies solely
on massively parallel local operations, avoiding global synchronization and
enabling efficient GPU implementation. Experiments demonstrate real-time,
large-scale simulations with performance more than two orders of magni-
tude faster than prior methods while maintaining consistent computational
budgets.

# OGC

This is my attempt at a physics engine in CUDA based on the research paper mentioned above.
