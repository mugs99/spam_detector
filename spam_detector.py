import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# importing the data
file_path = 'PATH HERE'
df = pd.read_csv(file_path, sep='\t', names=["label", "message"])
print(df.head())

# Data cleaning and preprocessing
lemmatizer = WordNetLemmatizer()
corpus = []

for i in range(len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['message'][i])  # Fixing the regex pattern
    review = review.lower().split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# creating the bag of words model
tv = TfidfVectorizer()
X = tv.fit_transform(corpus).toarray()

y = pd.get_dummies(df['label'])
y = y.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Training model using Naive bayes classifier
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred = spam_detect_model.predict(X_test)

# Evaluation
matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", matrix)
print("Accuracy: %.2f%%" % (accuracy*100))