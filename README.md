# Dense Neural Networks Are Not Universal Approximators

This repository contains the code used for the MNIST experiments in  
**“Dense Neural Networks Are Not Universal Approximators”**.

We empirically compare standard fully connected one-hidden-layer ReLU networks
with *dense* (weight-constrained) networks, and demonstrate that increasing width
in dense networks leads to early performance saturation, consistent with the
theoretical predictions in the paper.

---

## Overview

We train one-hidden-layer ReLU MLPs on MNIST under two regimes:

- **Standard MLP**: unconstrained weights, trained with Adam.
- **Dense MLP**: weights are clamped after each optimizer step to enforce
  per-layer bounds of the form  
  \[
  W_\ell \in [-B/d_{\ell-1},\, B/d_{\ell-1}],
  \]
  where \(d_{\ell-1}\) is the input dimension of the layer.

We vary the hidden-layer width from very small to highly overparameterized regimes
and evaluate final training and test accuracy across multiple random seeds.

---

## Repository Structure

