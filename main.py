from sklearn.datasets import fetch_20newsgroups
from typing import Any

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

newsgroup_data:Any = fetch_20newsgroups(subset="train", data_home="./data_sets")
X, y = newsgroup_data.data, newsgroup_data.target


nltk.download("punkt", download_dir="./venv/nltk_data")
nltk.download("stopwords", download_dir="./venv/nltk_data")

def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return " ".join(stemmed_tokens)

X_preprocessed = [preprocess_text(text) for text in X]

vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X_preprocessed)

classifier = MultinomialNB()
classifier.fit(X_transformed, y)

test_data:Any = fetch_20newsgroups(subset="test")
X_test, y_test = test_data.data, test_data.target

X_test_preprocessed = [preprocess_text(text) for text in X_test]
X_test_transformed = vectorizer.transform(X_test_preprocessed)

y_pred = classifier.predict(X_test_transformed)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
