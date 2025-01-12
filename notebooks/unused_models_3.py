import numpy as np
import pandas as pd
import gzip
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.sparse import hstack
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
turkish_stopwords = stopwords.words('turkish')

# Load Data
train_classification_df = pd.read_csv("train-classification.csv")
train_classification_df = train_classification_df.rename(columns={'Unnamed: 0': 'user_id', 'label': 'category'})
train_classification_df["category"] = train_classification_df["category"].apply(str.lower)
username2_category = train_classification_df.set_index("user_id").to_dict()["category"]

# Read JSONL Data
train_data_path = "training-dataset.jsonl.gz"
username2posts_train = {}
username2profile_train = {}

with gzip.open(train_data_path, "rt") as fh:
    for line in fh:
        sample = json.loads(line)
        profile = sample["profile"]
        username = profile["username"]
        if username in username2_category:
            username2posts_train[username] = sample["posts"]
            username2profile_train[username] = profile

# Preprocess Captions
def preprocess_text(text):
    if text is None:
        return ""
    text = text.casefold()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zçğıöşü0-9\s#@]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Prepare Data for Regression
corpus = []
profile_features = []
y_train = []

for username, posts in username2posts_train.items():
    cleaned_captions = []
    like_counts = []
    profile = username2profile_train[username]

    for post in posts:
        caption = post.get("caption", "")
        like_count = post.get("like_count", 0) or 0
        cleaned_captions.append(preprocess_text(caption))
        like_counts.append(like_count)

    if like_counts:
        y_train.append(np.mean(like_counts))  # Average like count for each user
    else:
        y_train.append(10.0)  # Default for users with no posts

    corpus.append(" ".join(cleaned_captions))
    followers = profile.get("follower_count", 0)
    following = profile.get("following_count", 1)
    post_count = len(posts)
    follow_ratio = followers / (following if following > 0 else 1)
    avg_caption_len = np.mean([len(c) for c in cleaned_captions]) if cleaned_captions else 0
    profile_features.append([followers, following, post_count, follow_ratio, avg_caption_len])

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words=turkish_stopwords, max_features=5000)
x_post_train = vectorizer.fit_transform(corpus)
profile_features = np.array(profile_features)

# Impute NaN values in profile features
profile_features = np.nan_to_num(profile_features)

x_combined_train = hstack([x_post_train, profile_features])

# Log Transformation to Reduce Skewness
y_train = np.log1p(y_train)

# Outlier Detection and Removal
q1, q3 = np.percentile(y_train, [25, 75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
valid_indices = (y_train >= lower_bound) & (y_train <= upper_bound)

x_combined_train = x_combined_train.tocsr()
x_combined_train = x_combined_train[valid_indices]
y_train = y_train[valid_indices]

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = []
mae_scores = []
r2_scores = []

for train_index, val_index in kf.split(x_combined_train):
    x_train, x_val = x_combined_train[train_index], x_combined_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.005, max_depth=10, random_state=42)
    model.fit(x_train, y_train_fold)
    y_val_pred = model.predict(x_val)
    mse_scores.append(mean_squared_error(y_val_fold, y_val_pred))
    mae_scores.append(mean_absolute_error(y_val_fold, y_val_pred))
    r2_scores.append(r2_score(y_val_fold, y_val_pred))

print(f"Cross-Validation MSE: {np.mean(mse_scores)}")
print(f"Cross-Validation MAE: {np.mean(mae_scores)}")
print(f"Cross-Validation R^2: {np.mean(r2_scores)}")

# Final Model Training
model.fit(x_combined_train, y_train)

# Test Data Prediction
path = "test-regression-round2  .jsonl"
output_path = "sample-regression-output.json"
output_list = []

with open(path, "rt") as fh:
    for line in fh:
        sample = json.loads(line)
        username = sample["username"]
        if username in username2posts_train:
            captions = " ".join([preprocess_text(p.get("caption", "")) for p in username2posts_train[username]])
            x_test_tfidf = vectorizer.transform([captions])
            profile = username2profile_train[username]
            follow_count = profile.get("following_count", 1)
            if follow_count == 0:
                follow_count = 1
            avg_caption_len = np.mean([len(preprocess_text(p.get("caption", ""))) for p in username2posts_train[username]])
            x_test_profile = np.array([[profile.get("follower_count", 0),
                                        follow_count,
                                        len(username2posts_train[username]),
                                        profile.get("follower_count", 0) / follow_count,
                                        avg_caption_len]])
            x_test = hstack([x_test_tfidf, x_test_profile])
            pred_like_count = np.expm1(model.predict(x_test)[0])
        else:
            pred_like_count = 10.0  # Default if user is unknown
        sample["like_count"] = max(0, int(pred_like_count))
        output_list.append(sample)

with open(output_path, "wt") as of:
    json.dump(output_list, of, indent=4)

print("Test predictions saved to sample-regression-output.json")
