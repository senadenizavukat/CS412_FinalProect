# CS412_FinalProect

## **Overview of the Repository**
This repository contains the scripts, data, and resources used for the [CS412_FinalProect]. The project focuses on [category classification of the profile] and [like-count prediction oer post].

### **Repository Structure**
- **`data/`**
  - Contains the datasets used for training and testing.
    - `training-dataset.jsonl.gz`: Training data.
      - **Profile Meta-Data**: Metadata for each account, including follower count, biography, and other account details.
      - **Most Recent Posts**: The 36 most recent posts for each account, used to extract features for classification.
    - `test-classification-roundx.dat`: Test data for each round.

- **`src/`**
  - Includes 3 main parts of the project.
    - Data Preparation : data cleaning, data mapping, TF-IDF feature extraction, and scaling.
    - Model Training: Implements XGBoost model training with hyperparameter tuning.
    - Evaluation: Includes model evaluation metrics like precision, recall, and F1-score.

- **`notebooks/`**
  - Jupyter notebooks for exploratory data analysis (EDA) and experimentation.

- **`results/`**
  - Stores the experimental outputs, including figures, tables, and metrics.

- **`README.md`**
  - This file, providing an overview of the project.

---


## **Methodology**
This project follows a systematic approach to classify [user profiles by category]. Key steps include:

### **Data Preprocessing**
   - **POST DATA:**
     1. **Timestamp is turned into Hour - Month Day - Week Day**
        
     <img width="765" alt="Ekran Resmi 2025-01-12 18 30 13" src="https://github.com/user-attachments/assets/ecbe4a00-adc4-4841-9fe4-7b681c5d5724" />

     2. **Hours/Like Count interaction**
        
     <img width="897" alt="Ekran Resmi 2025-01-12 18 33 52" src="https://github.com/user-attachments/assets/79794b64-23a6-40c9-b01e-04b7ccd811ee" />

     3. **Correlation of extended(^2, ^3, log) features**
        
      <img width="525" alt="Ekran Resmi 2025-01-12 19 14 22" src="https://github.com/user-attachments/assets/a36f4817-7ea3-4441-aaf2-d8d82143d910" />
      
    4. Check for outliers in the like count
      <img width="598" alt="Ekran Resmi 2025-01-12 19 34 56" src="https://github.com/user-attachments/assets/40361fb3-3b0f-42cf-94a4-df5f653aa873" />
    
    5.Outliers removed by using Interquartile Range Adjustment
    
    <img width="652" alt="Ekran Resmi 2025-01-12 19 36 30" src="https://github.com/user-attachments/assets/8dbe3bc3-38a9-415a-8cf3-b4bd84fa3f1a" />


   - **PROFILE DATA:**
     1. Unnecessary columns such as url columns, contact information and name columns are dropped
        
     2. Columns with single unique value are dropped
      <img width="933" alt="Ekran Resmi 2025-01-12 19 21 53" src="https://github.com/user-attachments/assets/9ff60222-ddf6-4be3-afb0-3d73800e6090" />

     3.  Numerical and Boolean Columns are turned from object datatype into int and float datatypes
        
     4. Categorical features are transformed using mapping and one-hot encoding.
        
     <img width="716" alt="Ekran Resmi 2025-01-12 19 16 23" src="https://github.com/user-attachments/assets/6714463e-d300-4395-863c-216cb9c15df8" />

     5. Extracted text features from user biographies and captions using TF-IDF.
        
     6. Scaled the dataset using MaxMin Scaler.
    
        




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
![Data Cleaning Process](images/Ekran Resmi 2025-01-12 18.21.16.png?raw=true)




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

