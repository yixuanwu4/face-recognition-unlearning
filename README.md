# face-recognition-unlearning

A lightweight facial recognition system with support for machine unlearning, allowing removal of a specific user without full model retraining.

## Overview

This project explores how to remove a user's influence from a trained model. Instead of retraining from scratch, the model is updated using only retained data with controlled noise injection.
The goal is to approximate the "right to be forgotten" in a practical and low-cost way.

## Method

The pipeline consists of:

1. Train a face recognition model on multiple users
1. Select a user to remove
1. Fine-tune the model on remaining data only
1. Apply:
  - gradient clipping
  - Gaussian noise on gradients

This weakens the model's ability to recognize the removed user while preserving performance on others.

## Results
- Forgotten user accuracy: close to random
- Retained users accuracy: largely preserved
- Unlearning cost: significantly lower than full retraining

## Notes

This is an experimental implementation of machine unlearning inspired by recent research.
It does not guarantee full data removal or legal compliance.

## Reference

Certified Unlearning for Neural Networks (Koloskova et al., 2025)