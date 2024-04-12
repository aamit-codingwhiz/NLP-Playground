# Fake Real News Classification

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Reading the dataset
fake_real_df = pd.read_csv('Fake_Real_Data.csv')

# Adding a numerical label column
label_mapping = {'Fake': 0, 'Real': 1}
fake_real_df['label_num'] = fake_real_df['label'].map(label_mapping)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    fake_real_df['Text'],
    fake_real_df['label_num'],
    test_size=0.2,
    random_state=42,
    stratify=fake_real_df['label_num']
)

# Attempt 1: KNN with Euclidean distance
pipeline_knn_euclidean = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1,3))),
    ('classifier', KNeighborsClassifier(n_neighbors=10, metric='euclidean'))
])
pipeline_knn_euclidean.fit(X_train, y_train)
y_pred_knn_euclidean = pipeline_knn_euclidean.predict(X_test)
print("Classification Report for KNN with Euclidean distance:\n", classification_report(y_test, y_pred_knn_euclidean))

# Attempt 2: KNN with Cosine distance
pipeline_knn_cosine = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1,3))),
    ('classifier', KNeighborsClassifier(n_neighbors=10, metric='cosine'))
])
pipeline_knn_cosine.fit(X_train, y_train)
y_pred_knn_cosine = pipeline_knn_cosine.predict(X_test)
print("Classification Report for KNN with Cosine distance:\n", classification_report(y_test, y_pred_knn_cosine))

# Attempt 3: RandomForest
pipeline_rf = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1,3))),
    ('classifier', RandomForestClassifier())
])
pipeline_rf.fit(X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)
print("Classification Report for RandomForest:\n", classification_report(y_test, y_pred_rf))

# Attempt 4: Multinomial Naive Bayes
pipeline_nb = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1,3))),
    ('classifier', MultinomialNB(alpha=0.75))
])
pipeline_nb.fit(X_train, y_train)
y_pred_nb = pipeline_nb.predict(X_test)
print("Classification Report for Multinomial Naive Bayes:\n", classification_report(y_test, y_pred_nb))

# Preprocessing text data
import spacy
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text)
    filtered_tokens = [token.lemma_ for token in doc if not (token.is_stop or token.is_punct)]
    return " ".join(filtered_tokens)

fake_real_df['preprocessed_text'] = fake_real_df['Text'].apply(preprocess_text)

# Retraining the models with preprocessed text data
X_train_preprocessed, X_test_preprocessed, _, _ = train_test_split(
    fake_real_df['preprocessed_text'],
    fake_real_df['label_num'],
    test_size=0.2,
    random_state=42,
    stratify=fake_real_df['label_num']
)

# Attempt 1: RandomForest with preprocessed text
pipeline_rf_preprocessed = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(3,3))),
    ('classifier', RandomForestClassifier())
])
pipeline_rf_preprocessed.fit(X_train_preprocessed, y_train)
y_pred_rf_preprocessed = pipeline_rf_preprocessed.predict(X_test_preprocessed)
print("Classification Report for RandomForest with preprocessed text:\n", classification_report(y_test, y_pred_rf_preprocessed))

# Attempt 2: RandomForest with preprocessed text and n-grams
pipeline_rf_preprocessed_ngrams = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1,3))),
    ('classifier', RandomForestClassifier())
])
pipeline_rf_preprocessed_ngrams.fit(X_train_preprocessed, y_train)
y_pred_rf_preprocessed_ngrams = pipeline_rf_preprocessed_ngrams.predict(X_test_preprocessed)
print("Classification Report for RandomForest with preprocessed text and n-grams:\n", classification_report(y_test, y_pred_rf_preprocessed_ngrams))

# Confusion matrix for the best model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_rf_preprocessed_ngrams)
print("Confusion Matrix:\n", cm)
