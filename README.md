# CS412_FinalProject

## **Overview of the Repository**
This repository contains the scripts, data, and resources used for the [CS412_FinalProect]. The project focuses on [category classification of the profile] and [like-count prediction oer post].

### **Repository Structure**
- **`data/`**
  - Contains the datasets used for training and testing.
    - `training-dataset.jsonl.gz`: Training data.
      - **Profile Meta-Data**: Metadata for each account, including follower count, biography, and other account details.
      - **Most Recent Posts**: The 36 most recent posts for each account, used to extract features for classification.
    - `test-classification-roundx.dat`: Test data for each round.


- **`notebooks/`**
  - Jupyter notebook that contains:
    - Data Preparation : data cleaning, data mapping, TF-IDF feature extraction, and scaling.
    - Model Training: Implements XGBoost model training with hyperparameter tuning.
    - Evaluation: Includes model evaluation metrics like precision, recall, and F1-score.
  
  - 2 more notebooks that contain data processing tools, libraries and machine learning models that were unused in the final product

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

      <img width="994" alt="Ekran Resmi 2025-01-12 23 35 49" src="https://github.com/user-attachments/assets/957c8f7c-e75b-4f10-8824-0870c3133fea" />

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
The classification component of the project focuses on categorizing user profiles into predefined categories using textual and numerical features extracted from profile metadata and posts. The key steps in this component are:

1. **Data Preprocessing**

**Profile Data:**

  1. Feature Selection: Removed irrelevant columns such as URLs and contact information to avoid introducing noise into the model.
  2. Data Type Conversion: Ensured that numerical and boolean columns were properly converted to their respective data types (e.g., int, float), facilitating seamless integration into the pipeline.
  3. Textual Feature Extraction:
  Used TF-IDF vectorization to extract textual features separately from user biographies and post captions.
  This approach created a numerical representation of text data while preserving the importance of each term in the context of the document.
  Tried models like BERT and FastText for more efficient word embedding-based classification. These models help improve the quality of predictions by capturing different aspects of the data. However, because of technical incapabilities we could not continue with these models.
  5. Data Mapping and Encoding:
  Mapped metadata such as user activities or preferences to meaningful numerical categories.
  Applied one-hot encoding to categorical features, enabling the model to process these variables effectively.
  6. Feature Scaling: Normalized numerical features using Min-Max Scaling to align their values within the same range, improving model convergence      and prediction consistency.
  
2. **Feature Merging**
  1. Textual Feature Integration:
  Combined the TF-IDF matrices for user biographies and captions into a single feature matrix. This ensured that textual information from both       sources was jointly utilized for classification.
  2. Unified Dataset Creation:
  Merged the combined TF-IDF matrix with encoded metadata and numerical features to create a holistic representation of the data.
  Feature selection through correlation analysis ensured that only the most impactful features were retained, reducing dimensionality and improving     computational efficiency.

**Correlation Analysis:** Encoded features show higher correlation compared to other features. 

<img width="403" alt="Ekran Resmi 2025-01-12 23 05 52" src="https://github.com/user-attachments/assets/959e6706-ee7f-495d-9df0-919eb01e9990" />

Correlation Analysis of Merged DataFrame: 

<img width="387" alt="Ekran Resmi 2025-01-12 23 07 26" src="https://github.com/user-attachments/assets/47f7cdf8-9e9d-4636-b070-3c7426f3bcfd" />


2. **Model Training**

  - After experimenting with multiple models, including Naive Bayes, linear, polynomial, and RBF SVM, as well as logistic regression, we trained the classification model using the RandomForestClassifier from scikit-learn. To optimize the performance of the Multinomial Naive Bayes model, we conducted a Grid Search over the alpha parameter (smoothing parameter). The grid search identified the optimal alpha, ensuring the best trade-off between overfitting and underfitting, as shown below:

  - To optimize the model's performance, conducted a Grid Search over the alpha parameter (smoothing parameter):

  <img width="452" alt="Ekran Resmi 2025-01-12 23 13 20" src="https://github.com/user-attachments/assets/7a12d209-6ec7-427f-a133-d7d45c5246e9" />
  
  - The grid search identified the optimal alpha parameter, ensuring the best trade-off between overfitting and underfitting:




3. **Evaluation**
The model was evaluated on both the training and validation datasets.

3.1 *Training Data Results:*
Accuracy: 99.7%
Macro and Weighted Average F1-scores: 1.00
Observations: The high performance on training data suggests overfitting.

3.2 *Validation Data Results:*
Accuracy: 59.2%
Macro Average Precision: 52%
Observations: Performance varied across categories, indicating overfitting and a need for better generalization.

3. **Evaluation**
The model was evaluated on both the training and validation datasets.


4. **Category-Wise Performance (Validation Data)**
<img width="713" alt="Screenshot 2025-01-12 at 22 04 48" src="https://github.com/user-attachments/assets/11e3a178-b87b-4eec-8606-e6d6f78a0745" />

5. **Test Data Prediction**
The trained model was applied to test data. Text features were transformed using the trained TF-IDF vectorizer, and predictions were generated for each profile.
Outputs were saved in a JSON file for further analysis.

6. **Challenges and Solutions**
Overfitting: Observed high accuracy on training data but relatively low generalization on validation data.
Solution: Implemented hyperparameter tuning (e.g., adjusting max_depth, min_samples_split) and explored ensemble methods.
Imbalanced Classes: Addressed using techniques like SMOTE to oversample minority classes.




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











## **Results**
The project achieved significant improvements in classification metrics, particularly for underrepresented classes. Below are the highlights:

**Classification**
- **Overall Accuracy**: 58%
- **Macro Average Precision**: 56%
- **Category-wise Performance**:
  - `food`: Precision: 75%, Recall: 91%, F1-score: 82%
  - `fashion`: Precision: 60%, Recall: 69%, F1-score: 64%

**Regression**
- **Overall Accuracy**: 0.59
- **Log Mean Squared Error**: 1053.998574

### **Figures**
![Confusion Matrix](results/confusion_matrix.png)

### **Tables**

![image](https://github.com/user-attachments/assets/38d06da4-8cf8-4cfa-8bec-23409c7043e2)


The classification results shown in the above screenshot highlight the significant impact of data distribution on model performance. Classes with a higher number of samples in the training dataset, such as "fashion," "gaming," and "health and lifestyle," performed markedly better in terms of precision, recall, and F1-score. This trend is evident from their higher metrics across the board, indicating that the model could effectively learn patterns and generalize well for these classes.

In contrast, classes with fewer samples, such as "art" and "entertainment," exhibited poorer performance, particularly in the validation set. The metrics for these underrepresented classes, especially recall and F1-score, were substantially lower, indicating difficulty in predicting these categories correctly. This discrepancy can be attributed to the imbalance in the dataset, where limited training examples for certain classes constrain the model's ability to recognize their distinct patterns.

Overall, these results emphasize the need for addressing class imbalance in the dataset to improve the performance of minority classes. Potential solutions include oversampling underrepresented classes, undersampling dominant classes, or using class weights during model training to ensure more equitable representation and better generalization across all categories.


---

## **Team Contributions**

| **Team Member**      | **Contributions**                                                                 |
|-----------------------|-----------------------------------------------------------------------------------|
| [Sena Deniz Avukat]             | Data preprocessing, feature engineering, and classification/like-prediction models.                |
| [Metin ulaş Erdoğan]             | Model development, hyperparameter tuning with GridSearchCV, and model evaluation. |
| [Adahan Yiğitol]             | Visualization and results documentation, including confusion matrix and metrics.  |
| [Barbaros Yahya]             | Repository structuring and integration of experimental findings into the report.  |
| [İpek Öke]             | Model development, feature engineering, data preprocessing and model evaluation and enhancements for classification.  |
| [Damla Salman]             | Report construction and analysis.|

