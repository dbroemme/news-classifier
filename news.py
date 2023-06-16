import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Read the CSV file
df = pd.read_csv('./data/news/news.csv', header=0)

# Convert the label column to a categorical variable
df['label'] = df['label'].astype('category')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.25)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the training and test data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create a logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train_tfidf, y_train)

# Predict the labels for the test data
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print(model.score(X_test_tfidf, y_test))

# Create a function to classify text as fake or real
def classify_text(text):
    # Transform the text data
    text_tfidf = vectorizer.transform([text])

    # Get the predicted label for the text data
    label = model.predict(text_tfidf)[0]

    # Return the label
    return label


# Open the text file in read mode
with open("article2.txt", "r") as f:
    # Read the text file into a variable
    text_1 = f.read()

# Print the text
print(classify_text(text_1))