# Emotion Detection

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import spacy

# Load data
df = pd.read_csv('Emotion_classify_Data.csv')
label_mapping = {'joy': 0, 'fear': 1, 'anger': 2}
df['Emotion_num'] = df['Emotion'].map(label_mapping)

# Preprocess text
nlp = spacy.load("en_core_web_sm")
def preprocess_text(text):
    doc = nlp(text)
    filtered_tokens = [token.lemma_ for token in doc if not (token.is_stop or token.is_punct)]
    return " ".join(filtered_tokens) 
df['preprocessed_comment'] = df['Comment'].apply(preprocess_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['preprocessed_comment'],
    df['Emotion_num'],
    test_size=0.2,
    random_state=2022,
    stratify=df['Emotion_num']
)

# Attempt 1
pipeline = Pipeline([
    ('countVectorizer', CountVectorizer(ngram_range=(1, 2))),
    ('RandomForestClassifier', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("Attempt 1 - Classification Report:\n", classification_report(y_test, y_pred))

# Attempt 2
pipeline = Pipeline([
    ('TfidfVectorizer', TfidfVectorizer()),
    ('MultinomialNB', MultinomialNB())
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("Attempt 2 - Classification Report:\n", classification_report(y_test, y_pred))

# Attempt 3
pipeline = Pipeline([
    ('countVectorizer', CountVectorizer(ngram_range=(1, 2))),
    ('RandomForestClassifier', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("Attempt 3 - Classification Report:\n", classification_report(y_test, y_pred))

# Attempt 4
pipeline = Pipeline([
    ('TfidfVectorizer', TfidfVectorizer()),
    ('RandomForestClassifier', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("Attempt 4 - Classification Report:\n", classification_report(y_test, y_pred))
