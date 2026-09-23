# Machine Learning Methods based on Radial Basis Functions

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

**Bachelor's Thesis in Mathematics for Engineering**  
*Politecnico di Torino* | *A.Y. 2025/2026*  
**Author:** Francesco Foglia  
**Advisor:** Prof. Tommaso Vanzan  

## Overview
This repository contains the code and numerical experiments developed for my Bachelor's thesis. 
The project explores the application of **Radial Basis Function (RBF) Neural Networks** for supervised learning tasks, specifically non-parametric regression and 2D spatial classification. 

Instead of relying on non-linear optimization techniques (e.g., Gradient Descent) which are computationally heavy and prone to local minima, this work formulates RBF networks as **strictly convex linear statistical models**. This approach guarantees global convergence, ensures numerical stability, and drastically reduces computational training time.

## Key Features & Algorithms
- **Linear RBF Networks:** Exact analytical resolution of network weights via dense linear system inversion, bypassing standard backpropagation.
- **Ridge Regression (Global & Local):** Implementation of Tikhonov regularization to manage the bias-variance tradeoff and prevent overfitting on noisy datasets.
- **Automated Model Selection:** Dynamic hyperparameter tuning and structural evaluation using **Leave-One-Out CV (LOO-CV)** and **Generalized Cross-Validation (GCV)**.
- **Algorithmic Efficiency:** 
  - Implementation of **Forward Selection**.
  - Implementation of **Orthogonal Least Squares (OLS)** via Gram-Schmidt orthogonalization.
  - Development of the **Regularised Orthogonal Least Squares (ROLS)** algorithm, reducing the subset selection computational complexity from $\mathcal{O}(p^2)$ to $\mathcal{O}(p)$.
- **Deep Learning Benchmarking:** Direct comparison between the linear algebraic approach (NumPy/SciPy) and completely supervised non-linear baselines (PyTorch).

