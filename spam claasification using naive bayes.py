import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# loading the dataset
data = pd.read_csv(r"C:\Users\lenovo\OneDrive\Documents\spam.txt", sep='\t', header=None, names=['label', 'text'])
print(data['text'])
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# training the dataset
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# punctuation removal
data["column_name"] = data["text"].map(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
print(data["column_name"])
# converting to lowercase
data["lower"] = data["column_name"].str.lower()
print(data["lower"])

# Removing stopwords
countve = CountVectorizer(stop_words='english')
cdf = countve.fit_transform(data["lower"])
bow = pd.DataFrame(cdf.toarray(), columns=countve.get_feature_names_out())
print(bow)

# vectorization
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vec, y_train)

# Make predictions on the testing set
y_pred = nb_classifier.predict(X_test_vec)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
