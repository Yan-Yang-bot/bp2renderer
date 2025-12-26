## Backpropagation to Reaction–Diffusion Systems  
*(Gray–Scott as an example)*

This repository contains preliminary experimental code exploring **gradient-based parameter learning in reaction–diffusion (RD) systems**, using the Gray–Scott model as a concrete example.

The project is based on my Zenodo technical note [1], which discusses two general approaches to task-driven optimization of RD systems via backpropagation. While that note was later extended into a broader research proposal in the context of a PhD application, **this repository focuses specifically on an early implementation of truncated Backpropagation Through Time (truncated BPTT)** through unrolled simulation steps.

At the current stage, the code is intended to:

- study the numerical and optimization behavior of truncated BPTT in RD systems while matching target patterns (no downstream tasks yet),
- examine stability and gradient flow issues when backpropagating through time-stepped PDE simulators,
- serve as a sandbox for validating feasibility before moving to more principled formulations.

In addition to truncated BPTT, the implementation includes an **adaptive learning-rate scaling mechanism** designed to reduce the risk of stepping into unstable regions that produce NaN outputs. There is also ongoing work that explores criteria inspired by numerical stability conditions (e.g., CFL-type constraints) to detect abnormal parameter updates **before** numerical breakdown occurs.

The **implicit differentiation via steady-state solutions** discussed in [1], as well as additional directions outlined in [2], are **not yet implemented here** and are planned for future work.

Collaboration, discussion, and mentoring are welcome.  
Feel free to contact me at **yan.yang.research@proton.me**.

---

### References

[1] Y. Yang, *Expectation-Maximization Style Algorithm for Task-Driven Differentiable Renderer Optimization*.  
Zenodo, Nov. 21, 2025.  
DOI: https://doi.org/10.5281/zenodo.17662717

[2] Y. Yang, *Towards PDE-Structured Generative Modeling: Differentiable Simulators, Flow-Matching, and Pattern Manifolds (Proposal)*.  
Zenodo, Dec. 11, 2025.  
DOI: https://doi.org/10.5281/zenodo.17897116
