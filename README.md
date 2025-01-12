# CS412_FinalProect
# **Project Title**: [Your Project Name]

## **Overview of the Repository**
This repository contains the scripts, data, and resources used for the [project name]. The project focuses on [brief description of the project’s goal or purpose].

### **Repository Structure**
- **`data/`**
  - Contains the datasets used for training and testing.
  - Example:
    - `training-dataset.jsonl.gz`: Training data.
    - `test-classification-round1.dat`: Test data.

- **`src/`**
  - Core scripts used for the project.
    - `data_preprocessing.py`: Handles data cleaning, TF-IDF feature extraction, and SMOTE-based oversampling.
    - `model_training.py`: Implements XGBoost model training with hyperparameter tuning.
    - `evaluation.py`: Includes model evaluation metrics like precision, recall, and F1-score.

- **`notebooks/`**
  - Jupyter notebooks for exploratory data analysis (EDA) and experimentation.

- **`results/`**
  - Stores the experimental outputs, including figures, tables, and metrics.

- **`README.md`**
  - This file, providing an overview of the project.

---

## **Methodology**
This project follows a systematic approach to classify [project-specific task or problem, e.g., user posts by category]. Key steps include:

1. **Data Preprocessing**
   - Handled missing values and transformed categorical features using one-hot encoding.
   - Extracted text features from user biographies and captions using TF-IDF.
   - Balanced the dataset using SMOTE to address class imbalance issues.

2. **Model Development**
   - Used **XGBoost** as the primary classification algorithm due to its robustness with tabular data.
   - Hyperparameter tuning via **GridSearchCV** to optimize performance.

3. **Evaluation**
   - Measured precision, recall, F1-score, and overall accuracy for all categories.
   - Addressed class imbalance by carefully monitoring macro and weighted metrics.

4. **Solutions Offered**
   - Improved minority class predictions by using SMOTE and weighting in the model.
   - Provided a comprehensive analysis of feature importance to understand the model’s decisions.

---

## **Results**
The project achieved significant improvements in classification metrics, particularly for underrepresented classes. Below are the highlights:

- **Overall Accuracy**: 58%
- **Macro Average Precision**: 56%
- **Category-wise Performance**:
  - `food`: Precision: 75%, Recall: 91%, F1-score: 82%
  - `fashion`: Precision: 60%, Recall: 69%, F1-score: 64%

### **Figures**
![Confusion Matrix](results/confusion_matrix.png)

### **Tables**

| **Category**           | **Precision** | **Recall** | **F1-Score** |
|-------------------------|---------------|------------|--------------|
| Art                    | 47%           | 21%        | 29%          |
| Entertainment           | 31%           | 34%        | 32%          |
| Food                   | 75%           | 91%        | 82%          |
| Gaming                 | 0%            | 0%         | 0%           |

---

## **Team Contributions**

| **Team Member**      | **Contributions**                                                                 |
|-----------------------|-----------------------------------------------------------------------------------|
| [Name 1]             | Data preprocessing, feature engineering, and SMOTE implementation.                |
| [Name 2]             | Model development, hyperparameter tuning with GridSearchCV, and model evaluation. |
| [Name 3]             | Visualization and results documentation, including confusion matrix and metrics.  |
| [Name 4]             | Repository structuring and integration of experimental findings into the report.  |

