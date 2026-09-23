# Machine Learning on Radial Basis Functions (RBF)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)

**BSc Thesis in Mathematical Engineering** — *Politecnico di Torino*

This repository contains the research, mathematical formulation, and experimental codebase for training Radial Basis Function (RBF) neural networks. The project explores treating RBF networks as strictly convex linear models, avoiding the non-convex optimization pitfalls (local minima) typical of standard deep learning approaches.

## 📖 Abstract & Methodology

Standard neural networks often suffer from highly non-convex loss landscapes. This thesis investigates an alternative approach using RBF networks for regression and 2D classification tasks. By fixing the non-linear hidden layer and optimizing only the output weights, the problem is reduced to a convex linear system.

Key algorithmic implementations include:
*   **ROLS (Regularised Orthogonal Least Squares):** An advanced algorithm implemented to reduce subset selection complexity to linear complexity, efficiently identifying the most significant centers for the RBF network.
*   **GCV (Generalized Cross-Validation):** An automated hyperparameter tuning method used to guarantee global convergence and prevent overfitting without requiring a separate validation set.
*   **Benchmarking:** The convex RBF approach is benchmarked against standard Deep Learning baselines implemented in **PyTorch** to compare training efficiency, convergence guarantees, and computational overhead.

## 📂 Repository Structure

*   `docs/`: Contains the full written thesis document (`Machine Learning on Radial Basis Function.pdf`) detailing the mathematical proofs, algorithmic complexity analysis, and extensive results.
*   `notebooks/`: Contains the interactive Jupyter Notebooks used for the experiments:
    *   `shockfunction.ipynb`: Demonstrates the RBF and ROLS algorithm applied to the regression of a highly non-linear "shock" function, showcasing the model's function approximation capabilities.
    *   `simulazioni_tesi.ipynb`: Contains the core experimental pipeline, including 2D classification tasks, automated hyperparameter tuning via GCV, and performance benchmarking against PyTorch deep learning models.
    *   
## 🚀 How to Run the Experiments

To reproduce the experiments and explore the algorithms, clone the repository and set up the Python environment:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/francescofoglia0/Machine-Learning-on-Radial-Basis-Function.git](https://github.com/francescofoglia0/Machine-Learning-on-Radial-Basis-Function.git)
   cd Machine-Learning-on-Radial-Basis-Function
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```
   Navigate to the `notebooks/` directory and open the `.ipynb` files to run the interactive experiments.

## 🎓 Academic Context

This research was conducted as the final thesis for the Bachelor's Degree in Mathematical Engineering at Politecnico di Torino (Graduated with 110/110 *cum laude*).
