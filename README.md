# Dense Neural Networks Are Not Universal Approximators

This repository contains the code for the MNIST experiments in  
**“Dense Neural Networks Are Not Universal Approximators.”**

We empirically compare standard fully connected one-hidden-layer ReLU networks
with *dense* (weight-constrained) networks and show that increasing width in dense
networks leads to early performance saturation, consistent with the theory.

---

## Overview

We train one-hidden-layer ReLU MLPs on MNIST under two regimes:

- **Standard MLP**  
  Fully connected network trained with Adam, no weight constraints.

- **Dense MLP**  
  After each optimizer step, weights are clamped to fixed bounds that scale
  inversely with the input dimension of each layer.

Hidden-layer width is varied from very small to highly overparameterized regimes.
Final training and test accuracy are recorded across multiple random seeds.

---

## Repository Structure

