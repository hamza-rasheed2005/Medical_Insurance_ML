# Medical Insurance Cost Prediction 🏥
### *End-to-End Linear Regression from Scratch*

This repository contains a professional implementation of a predictive model for medical insurance costs. The core engine of this project is built using **Vectorized Linear Regression**, following the rigorous mathematical framework provided by **Andrew Ng's Machine Learning Specialization**.

## 🎯 Project Objective
The goal is to predict individual medical costs billed by health insurance based on demographic and lifestyle data. This project moves beyond using "Black-Box" libraries by implementing the optimization algorithms manually to ensure a deep understanding of model convergence.

## 🛠️ Technical Implementation Details

### 1. Mathematical Foundation
* **Hypothesis Function**:  
  $$f_{w,b}(x) = w \cdot x + b$$
* **Vectorized Cost Function**:  
  Calculated using the **Squared Error Cost** function to measure the model's performance across the entire dataset.
* **Vectorized Gradient Descent**:  
  Developed an efficient optimization loop to update parameters $(\mathbf{w}, b)$ simultaneously using NumPy's matrix operations:
  - $w = w - \alpha \frac{\partial J(w,b)}{\partial w}$
  - $b = b - \alpha \frac{\partial J(w,b)}{\partial b}$

### 2. Engineering Workflow
* **Environment**: Developed and tested on **Fedora Workstation**.
* **Preprocessing**: 
    * Categorical encoding for non-numeric data (Smoker, Region, Sex) using `get_dummies`.
    * **Feature Scaling**: Utilized `StandardScaler` to normalize features, ensuring a spherical cost surface for faster and more stable Gradient Descent convergence.
* **Optimization**: Fine-tuned the learning rate ($\alpha = 0.01$) and iterations (1,000+) to achieve a stable global minimum.

## 📊 Results & Performance
* **Convergence**: The cost function showed a smooth exponential decay, validating the correctness of the vectorized gradient implementation.
* **Model Accuracy**: Achieved an **R² Score of 0.73+**, indicating that the model explains a significant portion of the variance in insurance charges.
* **Key Drivers**: Analysis of the learned weights ($\mathbf{w}$) confirmed that **Smoking Status** and **Age** are the primary predictors of higher medical costs.

## 📁 Project Structure
* `Medical_Insurance_ML.ipynb`: The main development notebook with step-by-step implementation.
* `medical_insurance.csv`: The dataset containing 2,772 records.
* `README.md`: Project documentation.

## 🚀 How to Run
1. Ensure you have a Python environment (preferably on Linux/Fedora).
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
Developed by Hamza Rasheed IT Engineering Student | AI & ML   
