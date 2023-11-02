from src.utils import get_tfidf_fake_news_dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the TF-IDF vectorizer and the training/testing data
tfidf_train, tfidf_test, tfidf_vectorizer, tfidf_y_train, tfidf_y_test = get_tfidf_fake_news_dataset()

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(tfidf_train, tfidf_y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(tfidf_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(tfidf_y_test, pred)
print("Accuracy Score:", score)

# Calculate the confusion matrix: cm
cm = confusion_matrix(tfidf_y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['FAKE', 'REAL'])
disp.plot()
plt.show()
