# 🏥 Predicting Diabetes Progression with an MLP Regressor  
This project demonstrates how to build a **Multilayer Perceptron (MLP)** using **Keras** to predict diabetes progression based on clinical data. The tutorial covers data preprocessing, model design, training with early stopping, and performance evaluation using meaningful metrics and engaging visualizations.  

---

## 🎯 **Objective**  
The objective of this project is to predict the progression of diabetes one year after baseline using a neural network-based regressor. The goals include:  
- Building a deep learning model to handle complex, nonlinear patterns in medical data  
- Implementing data scaling and splitting  
- Training the model with early stopping to prevent overfitting  
- Evaluating the model using regression-specific metrics and visualizations  

---

## 📂 **Dataset Overview**  
**Dataset:** Diabetes dataset from scikit-learn (`load_diabetes`)  
- **Samples:** 442  
- **Features:** 10 continuous numerical variables representing clinical measurements:  
    - `age` – Age of the patient  
    - `sex` – Gender of the patient  
    - `bmi` – Body Mass Index  
    - `bp` – Average blood pressure  
    - `s1-s6` – Blood serum measurements  
- **Target:** A continuous measure of diabetes progression one year after baseline  

**Target Range:** ~25 to ~346  

### 📊 **Data Splitting**  
- **Training set:** 353 samples (80%)  
- **Test set:** 89 samples (20%)  
- **Random State:** Ensured reproducibility  

### 📏 **Data Scaling**  
Scaling improves model performance by transforming data to zero mean and unit variance using `StandardScaler`.  

---

## 🏗️ **Model Overview**  
### **Model: Keras MLP Regressor**  
The model is built using a **Multilayer Perceptron (MLP)** with the following architecture:  
✅ Fully connected layers with ReLU activation  
✅ Batch Normalization to stabilize training  
✅ Dropout to prevent overfitting  
✅ Linear output for continuous regression  

### **Architecture**  
| Layer Type                  | Output Shape | Parameters |
|-----------------------------|--------------|------------|
| Dense(64, ReLU)              | (None, 64)    | 704        |
| BatchNormalization           | (None, 64)    | 256        |
| Dropout(0.2)                 | (None, 64)    | 0          |
| Dense(32, ReLU)              | (None, 32)    | 2080        |
| BatchNormalization           | (None, 32)    | 128         |
| Dropout(0.2)                 | (None, 32)    | 0           |
| Dense(16, ReLU)              | (None, 16)    | 528         |
| Dense(1, Linear)             | (None, 1)     | 17          |

**Total Parameters:** 3,713  

### 🧪 **Why MLP for Regression?**  
✔️ Handles complex, non-linear relationships  
✔️ Can approximate any continuous function  
✔️ Batch Normalization reduces covariate shift  
✔️ Dropout increases generalization  

---

## ⚙️ **Training and Early Stopping**  
### **Training Setup**  
- **Epochs:** 150  
- **Batch Size:** 32  
- **Optimizer:** Adam  
- **Loss Function:** Mean Squared Error (MSE)  
- **Validation Split:** 20%  
- **Early Stopping:**  
    - Monitors validation loss  
    - Stops training after 15 epochs of no improvement  

---

## 🚀 **Performance Metrics**  
### **Test Results:**  
| Metric               | Value |
|-----------------------|-------|
| **Mean Squared Error (MSE)** | 3410.114 |
| **R² Score**            | 0.356 (35.6% of variance explained) |

- **Moderate R²:** The model explains about 35.6% of the variation in progression.  
- **Relatively high MSE:** Indicates that some complexity in the data remains unexplained.  

---

## 📊 **Results and Visualizations**  
### 1. **Training vs. Validation Loss Curve**  
- Training loss decreases steadily  
- Validation loss initially decreases but diverges, showing mild overfitting  
- Early stopping prevents further divergence  

### 2. **Regression Plot: True vs. Predicted Values**  
- Shows positive correlation between true and predicted values  
- Scatter around the line shows variance the model didn't capture  

### 3. **Residual Plot**  
- Plots residuals (y_test - y_pred) against predicted values  
- No clear structure = no significant model bias  

### 4. **Violin Plot of Residuals**  
- Symmetric shape around zero = good distribution  
- High density around zero indicates low bias  

### 5. **Smoothed Learning Curve**  
- Rolling average smooths out noise  
- Convergence reached before early stopping  

---

## 🔎 **Lessons and Takeaways**  
✔️ Neural networks effectively capture complex, nonlinear relationships in medical data  
✔️ Batch normalization and dropout improve training stability and generalization  
✔️ Early stopping ensures training stops at the optimal point  
✔️ Tuning the number of hidden layers and neurons affects the trade-off between bias and variance  

---

## 💡 **Potential Improvements**  
🔹 **Hyperparameter Tuning:** Use KerasTuner to optimize layer size, dropout rates, and learning rate  
🔹 **Data Augmentation:** Generate synthetic features using polynomial combinations  
🔹 **Alternative Loss Function:** Mean Absolute Error (MAE) or Huber Loss for outlier sensitivity  
🔹 **Ensemble Learning:** Combine the MLP with Gradient Boosted Trees or SVR for increased accuracy  

---

## ⚙️ **How to Run the Project**  
### 1. **Clone the Repository**  
```bash
git clone https://github.com/your-username/diabetes-mlp-regressor.git
