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

1. **Data Preprocessing**
   - Handled missing values and transformed categorical features using one-hot encoding.
   - Extracted text features from user biographies and captions using TF-IDF.
   - Balanced the dataset using SMOTE to address class imbalance issues.
### **Steps in Data Cleaning**

#### **1. Loading the Data**
The raw dataset is loaded into a Pandas DataFrame for preprocessing. The dataset includes the following columns:
- `like_count`: Number of likes for a post.
- `comments_count`: Number of comments for a post.
- `hour`: The hour the post was created.
- `category`: The category of the post.
- `username`: The user who created the post.

**Script**:
```python
import pandas as pd

# Load the dataset
raw_data_path = "path_to_dataset.csv"
data = pd.read_csv(raw_data_path)

# Display the first few rows
print(data.head())
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

<img width="534" alt="Ekran Resmi 2025-01-12 18 25 17" src="https://github.com/user-attachments/assets/f12a36b0-988c-488a-b1d4-ce8523e4f383" />



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

