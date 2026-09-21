# SE4050 Deep Learning: Comparative Analysis of Deep Architectures for Pneumonia Detection

## Overview
This repository contains an end-to-end deep learning framework designed to detect pneumonia from chest X-ray images. We benchmark four distinct architectures under identical experimental conditions:
1. Baseline Custom Deep CNN
2. Residual Network (ResNet-50)
3. Compound Scaled CNN (EfficientNet-B0)
4. Vision Transformer (ViT-B/16)

## Repository Structure
- `data/`: Dataset acquisition scripts and preprocessing pipelines.
- `models/`: Architecture definitions and hyperparameter configs.
- `notebooks/`: Individual training runs with tracked random seeds.
- `results/`: Confusion matrices, ROC-AUC curves, and metric logs.
- `requirements.txt`: Python package dependencies for exact reproducibility.