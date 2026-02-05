# Backpropagation to Reaction–Diffusion Systems
*(Gray–Scott as an example)*

---

## How to use this code repo

💡 Create and activate a virtual environment using Python 3.9, run
```
pip install -r requirements.txt
```
---

💡 Make sure your device (CPU/GPU) uses float64 instead of float32:
```
python check_float64.py
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

## How This is Different from PINN

As I understand them, PINNs primarily adapt the model representation by incorporating physical constraints into the loss function, while still relying on standard backpropagation-based optimizers. (The field is moving fast, and I may be behind on newer variants.)

My idea is almost the opposite.
Instead of reshaping the PDE/ODE into a neural-network-like form, I try to keep the original dynamical system intact and experiment with modifying or customizing the optimizer itself, performing backward propagation directly through the PDE/ODE to search for parameters whose stable outcomes are closest to a target pattern.

At the moment, I’m honestly not sure whether one can design an optimizer strategy that is capable of traversing a highly rugged dynamical-system parameter landscape and reliably reaching such target parameter sets. Even if the resulting strategy ends up looking quite different from standard backpropagation, that would still be acceptable to me.

The viability question must be somehow discussed before I proceed. My background in dynamical systems is limited, and ChatGPT tends to respond optimistically regardless of feasibility.

The original motivation was to connect pattern-generating systems like Gray–Scott with downstream tasks in an end-to-end differentiable pipeline. I also have the intuition that, in terms of information flow, direct backpropagation is more efficient than alternative indirect approaches, which is why I started thinking along this direction in the first place.

---

## Project Description

This repository contains **early experimental code** on **gradient-based parameter learning in reaction–diffusion systems**, using the Gray–Scott model.

It implements a **preliminary truncated BPTT** scheme through unrolled Gray-Scott simulations, following part of the ideas from my Zenodo technical note [1]. The goal is to probe **numerical stability and gradient behavior** when backpropagating through time-stepped PDEs, without downstream tasks.

Other useful components include:

- an adaptive learning-rate scheme;

- an embedded animation tool

    -- visualizing simulation steps for any parameter set along the learning trajectory,

    -- ensuring that its output is identical with what produced during training.

    -- visualizing simulation steps for multiple parameter sets (e.g., from the same trajectory) to investigate a trajectory.

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

---

## Experiment Result (Preliminary)

Animations from left to right:

No.1 - Initial parameters Du=0.1270, Dv=0.1269, F=0.0500, k=0.0501.

No.2 - Initial training (using L2 loss on 2d power spectrum + learning rate 1.2e-2 + 8918 iterations),
now Du=0.1285, Dv=0.0734, F=0.0429, k=0.0682.

No.3 - Continued training after reaching No.2 (using L2 loss on 2d power spectrum + learning rate 1e-3 + 97 iterations),
now Du=0.1343, Dv=0.0679, F=0.0429, k=0.0653.

* So we have optimizer trajectory: No.1 -> No.2 -> No.3.

No.4 - Target animation with Du=0.16, Dv=0.08, F=0.035, k=0.065.

[![Watch the video](media/gray_scott.gif)](media/gray_scott.mp4)

Limitations:

- The generation mechanism of initial states (with limited randomness) is shared between target and training.
- Where to stop training had to be decided manually, because loss value almost constantly remained between 245.0~270.0,
very rarely showing a few under 20.0, providing limited information.
The stable states of the parameter sets at "under-twenty losses" are like the following:

<img src="media/four_dots.png" width="20%" height="20%">


The author is continuing the exploration.

