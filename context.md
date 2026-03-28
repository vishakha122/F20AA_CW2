# F20AA Coursework 2 - Project Context for Task 4

## Repository Structure

```
F20AA_CW2/
├── data/
│   ├── train.csv              # Original training data (288,000 rows -> 440,088 lines)
│   ├── test.csv               # Original test data (109,700 lines, has 'rating' column)
│   ├── train_english.csv      # English-filtered training data (271,897 rows, 413,309 lines)
│   └── test_english.csv       # English-filtered test data (69,907 rows, 104,977 lines)
├── Task1_EDA.ipynb            # Task 1: Data Exploration and Visualization
├── Task2_preprocessing.ipynb  # Task 2: Text Processing and Normalization
├── Task3_vector_space.ipynb   # Task 3: Vector Space Model and Feature Representation
├── report_Task1&2.docx        # Written report for Tasks 1 and 2
└── README.md
```

## Data Files

### Primary Training Data: `data/train_english.csv`
- **Rows**: 271,897 (filtered from 288,000 original)
- **Columns**: `text`, `rating`, `text_length`
- **Use this file** for Task 4 model training

### Primary Test Data: `data/test_english.csv`
- **Rows**: 69,907 (filtered from ~109,700 original)
- **Columns**: `text`, `rating`
- **Note**: This test set DOES contain labels (ratings), so it can be used for evaluation
- **Use this file** for Task 4 model evaluation / Kaggle submission predictions

### Column Names
- **Text column**: `text` (raw review text)
- **Label column**: `rating` (integer, 1-5 star ratings)

### Target Classes
- **5 classes**: 1, 2, 3, 4, 5 (star ratings)
- **Distribution in training set** (from cleaned English data):
  - Mean: 2.61, Median: 3.0, Mode: 1
  - Skewness: ~0.16 (slight positive skew)
  - Class 1 (1-star) is the most frequent class
  - Classes 2 and 3 are underrepresented relative to 1, 4, 5

## Task 1 Summary (Data Exploration)

### Cleaning Steps Applied
1. Removed reviews shorter than 20 characters (9,692 removed)
2. Removed duplicates (0 found)
3. Filtered to English-only using `langid` library (reduced to 271,897 train / 69,907 test)
4. Removed rows with no alphabetic characters
5. Removed null/empty text rows

### Key Findings
- Text length: median 173 chars, mean 315 chars, max 8049 chars
- Very short reviews (<20 chars): 9,692 removed
- Very long reviews (>500 chars): 52,025
- VADER sentiment scores correlate with ratings (higher ratings -> more positive)
- 1-star reviews dominated by words: never, told, time, said, called, service
- 5-star reviews dominated by words: great, recommend, highly, always, best

## Task 2 Summary (Text Processing)

### Preprocessing Pipeline (use this exact pipeline for Task 4)

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
stop_words -= {'not', 'never', 'no', 'nor', 'none'}  # Keep negation words
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
    text = re.sub(r'<.*?>', '', text)                    # Remove HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', '', text)   # Remove URLs
    text = text.lower()                                   # Lowercase
    text = handle_negations(text)                         # Handle negations
    text = re.sub(r'[^\w\s]', '', text)                  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]  # Remove stopwords (except negations)
    tokens = [lemmatizer.lemmatize(w) for w in tokens]   # Lemmatize
    return " ".join(tokens)
```

### Key Decisions
- **Negation words preserved**: `not`, `never`, `no`, `nor`, `none` kept in stopword removal
- **Lemmatization** used (not stemming)
- **Abbreviation expansion** was implemented in Task 2 but NOT used in Task 3's pipeline
  (Task 3 used the simpler version without abbreviation expansion)
- The preprocessing pipeline in Task 3 is the one to follow (simpler, proven effective)

### Task 2 Vectorization Results (baseline with LogisticRegression, max_features=5000)
| Method | Accuracy | F1 (weighted) |
|--------|----------|---------------|
| Bag of Words | 0.6498 | 0.6068 |
| TF-IDF Unigrams | 0.6583 | 0.6164 |
| TF-IDF Bigrams (1,2) | 0.6669 | 0.6277 |

## Task 3 Summary (Feature Representation)

### All Results (LogisticRegression, max_features=10000)

| Representation | Accuracy | F1 (weighted) |
|---|---|---|
| TF-IDF (1,3) trigrams | 0.6690 | 0.6324 |
| TF-IDF (1,2) bigrams | 0.6682 | 0.6309 |
| TF-IDF (1,1) unigrams | 0.6595 | 0.6194 |
| Binary (bigrams) | 0.6567 | 0.6252 |
| BoW (1,3) trigrams | 0.6547 | 0.6213 |
| BoW (1,2) bigrams | 0.6544 | 0.6211 |
| Binary (unigrams) | 0.6471 | 0.6113 |
| BoW (unigrams) | 0.6458 | 0.6085 |
| Word2Vec (300d) | 0.6419 | 0.5877 |
| Word2Vec (100d) | 0.6350 | 0.5752 |

### Best Feature Extraction Settings for Task 4
- **Method**: TF-IDF
- **ngram_range**: (1, 2) — bigrams recommended over trigrams (marginal gain not worth complexity)
- **max_features**: 10,000 (used in Task 3; can be tuned further in Task 4)
- **Baseline accuracy to beat**: ~0.6682 (LogisticRegression with TF-IDF bigrams)

### Key Insights
- TF-IDF consistently outperforms BoW and Binary representations
- Bigrams capture phrases like "not good", "highly recommend" — important for sentiment
- Trigrams gave negligible improvement over bigrams (+0.0008 accuracy)
- Word2Vec underperformed sparse methods (averaging loses word order/emphasis)
- Classes 2 and 3 are hardest to predict (F1 ~0.21 and ~0.28)
- Class 1 is easiest (recall 0.92 with strong negative language)

## Recommended Setup for Task 4

### Data Loading
```python
df_train = pd.read_csv('data/train_english.csv')
df_test  = pd.read_csv('data/test_english.csv')
```

### Feature Extraction
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train = tfidf.fit_transform(df_train['processed'])  # after applying preprocess()
X_test  = tfidf.transform(df_test['processed'])
y_train = df_train['rating']
y_test  = df_test['rating']
```

### Dependencies
- pandas, numpy, matplotlib, seaborn
- nltk (punkt, stopwords, wordnet)
- scikit-learn
- gensim (for Word2Vec, used in Task 3)

## Issues and Notes
1. **train_english.csv has an extra column** `text_length` that train.csv does not have — not a problem, just ignore it
2. **test_english.csv has labels** (`rating` column) — can be used for proper train/test evaluation
3. **No abbreviation expansion** in the Task 3 pipeline — keep it consistent; use the simpler preprocessing
4. **Class imbalance**: Consider using `class_weight='balanced'` in models, or stratified splitting
5. **Baseline to beat**: LogisticRegression + TF-IDF (1,2) = 0.6682 accuracy / 0.6309 F1
