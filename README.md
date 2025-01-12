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
      
     4. **Check for outliers in the like count**
      <img width="598" alt="Ekran Resmi 2025-01-12 19 34 56" src="https://github.com/user-attachments/assets/40361fb3-3b0f-42cf-94a4-df5f653aa873" />
    
     5. **Outliers removed by using Interquartile Range Adjustment**
    
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
    
        
## **Model Development : Classification**



## **Model Development : Regression**
This code offers several custom solutions for predicting like_count by insights derived from data.

1. **Uses Historical Data for Predictions**
The function calculates average like counts and average comment counts for a user’s past posts. This ensures predictions are personalized for each user based on their historical engagement metrics.

2. **Incorporates Hourly Impact**
Hourly coefficients derived from the hour-by-like count correlation graph adjust the prediction. These coefficients capture trends where posts made during certain hours perform better or worse (e.g., posts during prime hours like 16:00-22:00 receive more likes).
The hourly adjustment ensures that the prediction gets reward or penalty to its average like count according to the hour.

Posts during hours like **16:00–22:00** perform better.
Posts during non-peak hours like **3:00–7:00** perform worse.

By incorporating these coefficients, the function directly leverages this insight, resulting in more accurate predictions based on the time of posting.

4. **Penalizes Below-Average Engagement**
If a post’s comment count is below the user’s average, a penalty is applied to the prediction. This accounts for the fact that lower-than-expected engagement (in terms of comments) is often correlated with fewer likes.


5. **Logarithmic Adjustment for Comment Counts**
The log_comments_count feature is used to account for the non-linear impact of comment counts on like counts. This transformation reduces the influence of outliers and models diminishing returns as comments increase. Even though it doesn't imply a strong connection, it is better than non logarithmic correlation. 
<img width="633" alt="Ekran Resmi 2025-01-12 20 24 52" src="https://github.com/user-attachments/assets/f3325e20-2789-4e28-903d-2487137d3edf" />

6. **Fallback for Missing Data**
If information like the hour or comments_count is missing from the current_post, it retrieves this data from the user’s historical posts or the train_cleaned_profile_df. This ensures predictions are robust even in cases of incomplete data.

### Why Heuristic-Based Approach Instead of a Machine Learning Model?

This approach is straightforward and interpretable. Each adjustment is easy to understand and directly tied to specific patterns in the data. It doesn’t require extensive training or computational resources. It’s simpler to implement and debug while still leveraging important patterns in the data.

The use of user-specific averages ensures that predictions are tailored to individual users, which is harder to achieve with general-purpose machine learning models without introducing complexity.

Manually analyzing the correlation between features (e.g., hour vs. like count), allows  more targeted and meaningful adjustments than a machine learning model might capture without feature engineering.







5. **Evaluation**
   - Measured precision, recall, F1-score, and overall accuracy for all categories.
   - Addressed class imbalance by carefully monitoring macro and weighted metrics.

6. **Solutions Offered**
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

