# Face Recognition Unlearning

A lightweight face recognition system with an experimental unlearning pipeline for removing a specific identity without full retraining.

---

## Overview

This project explores a practical question:
**can a trained model reduce its reliance on a specific user after deployment, without being retrained from scratch?**

Instead of deleting data and retraining, the model is updated in-place using a small amount of additional training.

---

## Method

The workflow is intentionally simple and reproducible:

1. Train a baseline classifier on all users
2. Select one identity as the forget target
3. Apply a short unlearning phase:
    - train on retained users only
    - introduce conflicting signals for the target identity
    - inject controlled noise into gradients

The goal is to locally disrupt the model’s representation of the target user while preserving global performance.
---

## Results

A typical run shows:

- Baseline model
    - Forget user accuracy: 1.000
    - Test accuracy: 0.66
- After unlearning
    - Forget user accuracy: 0.43
    - Test accuracy: 0.63

This shows a clear drop in performance on the target identity, while overall accuracy remains relatively stable.
*Results vary across runs due to stochastic training dynamics.*
*The numbers above reflects consistent trends rather than exact reproducibility.*

---

## Insight

Unlearning here is not explicit deletion.
Instead, the model is pushed into a conflicting state where previously learned signals about the target identity are weakened.

In practice, this behaves as targeted degradation rather than complete removal.

---

## Limitations

- No guarantee of full data removal
- Sensitive to hyperparameters (noise level, learning rate, training length)
- Residual information about the target may remain
- Not suitable for compliance-critical or privacy-sensitive applications

---

## Notes

This is an experimental project focused on intuition and implementation.

It highlights both:

- how surprisingly easy it is to partially degrade a learned identity
- and how difficult true unlearning is in modern neural networks