# === Cell 0 ===
# install xgboost and lightgbm if not already available
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost', 'lightgbm', '-q'])

# === Cell 1 ===
pip3 install xgboost

# === Cell 2 ===
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# model selection and evaluation
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, precision_score, recall_score)

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb

# sparse matrix utilities
from scipy import sparse

# nltk data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

print("All libraries imported successfully.")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"XGBoost: {xgb.__version__}")
print(f"LightGBM: {lgb.__version__}")

# === Cell 3 ===
df_train = pd.read_csv('data/train_english.csv')
df_test  = pd.read_csv('data/test_english.csv')

print(f"Training set shape: {df_train.shape}")
print(f"Test set shape:     {df_test.shape}")
print(f"\nTraining columns: {list(df_train.columns)}")
print(f"Test columns:     {list(df_test.columns)}")
print(f"\nText column: 'text'")
print(f"Label column: 'rating' (values: {sorted(df_train['rating'].unique())})")
print(f"\nFirst 3 rows:")
df_train[['text', 'rating']].head(3)

# === Cell 4 ===
# preprocessing pipeline — identical to Task 3

stop_words = set(stopwords.words('english'))
stop_words -= {'not', 'never', 'no', 'nor', 'none'}

lemmatizer = WordNetLemmatizer()

def remove_emojis(text):
    emoji_pattern = re.compile(
        "[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF" u"\U0001FA70-\U0001FAFF" "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)

def handle_negations(text):
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    return text

def preprocess(text):
    text = remove_emojis(str(text))
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.lower()
    text = handle_negations(text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# quick sanity check
test_text = "I can't believe how TERRIBLE this place is! Never going back."
print(f"Original:  {test_text}")
print(f"Processed: {preprocess(test_text)}")

# === Cell 5 ===
print("Applying preprocessing to training set...")
df_train['processed'] = df_train['text'].apply(preprocess)
print("Applying preprocessing to test set...")
df_test['processed']  = df_test['text'].apply(preprocess)
print("Done.")

# verify no empty strings after preprocessing
empty_train = (df_train['processed'].str.strip() == '').sum()
empty_test  = (df_test['processed'].str.strip() == '').sum()
print(f"\nEmpty processed texts — train: {empty_train}, test: {empty_test}")

# === Cell 6 ===
X_full = df_train['processed']
y_full = df_train['rating']

X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full,
    test_size=0.2,
    random_state=42,
    stratify=y_full
)

print(f"Full training set:  {len(X_full):,} samples")
print(f"Train split:        {len(X_train):,} samples (80%)")
print(f"Validation split:   {len(X_val):,} samples (20%)")
print(f"Held-out test set:  {len(df_test):,} samples (separate file)")

# === Cell 7 ===
# verify stratification preserved class distribution
print("Class distribution comparison:\n")
print(f"{'Rating':<10} {'Full':>10} {'Train':>10} {'Val':>10}")
print("-" * 42)

for rating in sorted(y_full.unique()):
    full_pct  = (y_full == rating).sum() / len(y_full) * 100
    train_pct = (y_train == rating).sum() / len(y_train) * 100
    val_pct   = (y_val == rating).sum() / len(y_val) * 100
    print(f"{rating:<10} {full_pct:>9.1f}% {train_pct:>9.1f}% {val_pct:>9.1f}%")

# visualise the distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (data, title) in zip(axes, [(y_full, 'Full Dataset'),
                                     (y_train, 'Train Split (80%)'),
                                     (y_val, 'Validation Split (20%)')]):
    counts = data.value_counts().sort_index()
    ax.bar(counts.index, counts.values, color='steelblue', edgecolor='navy')
    ax.set_title(title)
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    ax.set_xticks([1, 2, 3, 4, 5])

plt.suptitle('Class Distribution — Stratified Split Verification', fontsize=13)
plt.tight_layout()
plt.show()

# === Cell 8 ===
# TF-IDF vectorizer with settings from Task 3
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)

# fit on training split only, then transform train, val, and held-out test
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)
X_test_tfidf  = tfidf.transform(df_test['processed'])

print(f"Training matrix:   {X_train_tfidf.shape}")
print(f"Validation matrix: {X_val_tfidf.shape}")
print(f"Test matrix:       {X_test_tfidf.shape}")

# === Cell 9 ===
# feature matrix statistics
total_elements = X_train_tfidf.shape[0] * X_train_tfidf.shape[1]
nonzero = X_train_tfidf.nnz
sparsity = (1 - nonzero / total_elements) * 100

print("=" * 55)
print("TASK 4 ENVIRONMENT SUMMARY")
print("=" * 55)

print(f"\n--- Data ---")
print(f"Training samples:    {X_train_tfidf.shape[0]:,}")
print(f"Validation samples:  {X_val_tfidf.shape[0]:,}")
print(f"Test samples:        {X_test_tfidf.shape[0]:,}")

print(f"\n--- Features ---")
print(f"Vocabulary size:     {len(tfidf.vocabulary_):,}")
print(f"Feature matrix:      {X_train_tfidf.shape}")
print(f"Non-zero entries:    {nonzero:,}")
print(f"Sparsity:            {sparsity:.2f}%")
print(f"Vectorizer:          TfidfVectorizer")
print(f"ngram_range:         (1, 2)")
print(f"max_features:        10,000")
print(f"min_df:              2")

print(f"\n--- Class Distribution (Train) ---")
for rating in sorted(y_train.unique()):
    count = (y_train == rating).sum()
    pct = count / len(y_train) * 100
    print(f"  Rating {rating}: {count:>7,} ({pct:.1f}%)")

print(f"\n--- Baseline to Beat ---")
print(f"LogReg + TF-IDF (1,2) from Task 3: Acc=0.6682, F1=0.6309")
print("=" * 55)

# === Cell 10 ===
# show some of the top TF-IDF features to sanity-check the vocabulary
feature_names = tfidf.get_feature_names_out()

# top features by average TF-IDF weight across all training documents
mean_tfidf = X_train_tfidf.mean(axis=0).A1
top_indices = mean_tfidf.argsort()[-20:][::-1]

print("Top 20 features by mean TF-IDF weight across training set:")
for i, idx in enumerate(top_indices, 1):
    print(f"  {i:2d}. {feature_names[idx]:<25} (mean weight: {mean_tfidf[idx]:.4f})")

# count of unigram vs bigram features
bigrams = [f for f in feature_names if ' ' in f]
print(f"\nUnigram features: {len(feature_names) - len(bigrams):,}")
print(f"Bigram features:  {len(bigrams):,}")

# === Cell 11 ===
# save the fitted TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("Saved: tfidf_vectorizer.pkl")

# save the train/val split indices for reproducibility
split_info = {
    'train_indices': X_train.index.tolist(),
    'val_indices': X_val.index.tolist(),
    'random_state': 42,
    'test_size': 0.2
}
with open('split_info.pkl', 'wb') as f:
    pickle.dump(split_info, f)
print("Saved: split_info.pkl")

print("\nEnvironment setup complete. Ready for model training.")

