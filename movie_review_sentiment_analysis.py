# Sentiment Analysis of Movie Reviews

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


movie_reviews_df = pd.read_csv('movies_sentiment_data.csv')

print("Shape of the dataset:", movie_reviews_df.shape)
print("Top 5 datapoints:\n", movie_reviews_df.head(5))
print("Distribution of sentiment:\n", movie_reviews_df['sentiment'].value_counts())

# Creating binary labels
movie_reviews_df['Category'] = movie_reviews_df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
print("Distribution of binary labels:\n", movie_reviews_df['Category'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(movie_reviews_df['review'], movie_reviews_df['Category'], test_size=0.2, random_state=42)

pipeline_rf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', RandomForestClassifier(n_estimators=50, criterion='entropy', verbose=1))
], verbose=True)
pipeline_rf.fit(X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)
print("Classification Report for Random Forest:\n", classification_report(y_test, y_pred_rf))

pipeline_knn = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', KNeighborsClassifier(n_neighbors=10, metric='euclidean'))
], verbose=True)
pipeline_knn.fit(X_train, y_train)
y_pred_knn = pipeline_knn.predict(X_test)
print("Classification Report for KNN:\n", classification_report(y_test, y_pred_knn))


pipeline_nb = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
], verbose=True)
pipeline_nb.fit(X_train, y_train)
y_pred_nb = pipeline_nb.predict(X_test)
print("Classification Report for Multinomial Naive Bayes:\n", classification_report(y_test, y_pred_nb))
