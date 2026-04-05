# Face Recognition Unlearning

A simple face recognition system with an experimental implementation of machine unlearning.

The project explores whether a trained model can "forget" a specific user without full retraining, by fine-tuning only on retained data and applying simple gradient-based perturbations.

---

## Overview

This project is an experiment in machine unlearning applied to face recognition.

Instead of retraining the model from scratch after removing a user’s data, we attempt to reduce the model’s dependence on that user by:

- continuing training on remaining users
- applying controlled noise during optimization
- discouraging correct classification of the removed identity

The goal is not perfect removal, but a measurable reduction in model confidence for the forgotten user.

---

## Method

The workflow is intentionally lightweight:

1. Train a baseline face recognition model on all users
2. Select one user as the "forget target"
3. Fine-tune the model using:
   - only retained user data
   - additional perturbation on gradients
   - weak supervision to discourage memorization of the removed identity

---

## Key Idea

Instead of explicitly deleting knowledge from the model, we try to **overwrite its decision boundary locally** using noisy and conflicting gradients.

This makes the model less confident about the removed user while preserving general recognition ability.

---

## Results (observed behavior)

In experiments:

- The model still performs well on retained users
- Performance on the forgotten user drops, but not to zero
- The effect varies depending on training stability and noise level

In practice, this behaves more like *partial forgetting* rather than complete removal.

---

## Limitations

This approach is intentionally simple and has several limitations:

- It does not guarantee full removal of user information
- Forgetting strength depends heavily on hyperparameters
- Some residual information about the removed user may remain in the model
- This is not a certified or privacy-guaranteed method

---

## Notes

This project is an educational exploration of machine unlearning techniques inspired by recent research.

It demonstrates that partial forgetting is possible with minimal changes to a standard training pipeline, but also highlights the difficulty of achieving true unlearning in practice.