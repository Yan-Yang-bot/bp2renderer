## How to use this code repo

💡 Make sure your device (CPU/GPU) uses float64 instead of float32.

---

💡 Create and activate a virtual environment using Python 3.9, run
```
pip install -r requirements.txt
```
---

💡 To start exploring the parameter space of Gray Scott, run
```
python trainer.py
```
There is a target generation stage the first time you run it, but the targets will be stored in a `*.pt` file and reused when you repeat this command. Entering training, logs will show you at each iteration what gradients are generated, what learning rates are tried, which learning rate is finally used, how many Gray Scott discretized steps were taken, when there are overflows or NaN, and what the current parameters are.

---

💡 If you're not satisfied with just log information, copy the logged parameters into `generate_targets.py` and use the following to show stepping animations using such parameters (a list of multiple parameters from a parameter search trajectory can be used for simultaneous animation, so that you can see where the optimizer is going):
```
python generate_targets.py 4
```

---

💡 To plot pair-wise losses (as heat maps) among generated targets in `*.pt`, use
```
python within_target_loss.py
```

---

## Backpropagation to Reaction–Diffusion Systems  
*(Gray–Scott as an example)*

This repository contains **early experimental code** on **gradient-based parameter learning in reaction–diffusion systems**, using the Gray–Scott model.

It implements a **preliminary truncated BPTT** scheme through unrolled RD simulations, following ideas from my Zenodo technical note [1]. The goal is to probe **numerical stability and gradient behavior** when backpropagating through time-stepped PDEs, without downstream tasks.

Other useful components include:

- an adaptive learning-rate scheme;

- an embedded animation tool

    -- visualizing forward simulation steps for any parameter set along the learning trajectory,

    -- ensuring that its output is identical with what produced during training.

**Next step:** the author will **systematically discuss how gradient-based optimizers can traverse different regions of the Gray–Scott parameter space**, including:

- **homogeneous steady states**  
  *(including trivial / collapsed solutions)*

- **pattern-forming steady states**

- **non-convergent but bounded dynamics**  
  *(oscillatory / quasi-periodic)*

- **chaotic spatiotemporal dynamics**  
  *(if verified, not numerical)*

- **fractal-like bifurcation boundaries**

*(The author plans to avoid going into **numerically invalid regions** strictly, so they are not in the above list.)*

These insights will guide further code updates and optimizer design.  
Implicit differentiation and other extensions discussed in [1,2] are not yet implemented.

Contact: **yan.yang.research@proton.me**


### References

[1] Y. Yang, *Expectation-Maximization Style Algorithm for Task-Driven Differentiable Renderer Optimization*.  
Zenodo, Nov. 21, 2025.  
DOI: https://doi.org/10.5281/zenodo.17662717

[2] Y. Yang, *Towards PDE-Structured Generative Modeling: Differentiable Simulators, Flow-Matching, and Pattern Manifolds (Proposal)*.  
Zenodo, Dec. 11, 2025.  
DOI: https://doi.org/10.5281/zenodo.17897116
