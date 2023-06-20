import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv("news.csv", header=0)

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Transform the text column into a sparse matrix of term frequencies
X = vectorizer.fit_transform(df["text"])

# Create a LogisticRegression model
model = LogisticRegression()

# Train the model on the features and labels
model.fit(X, df["label"])

print("Accuracy: " + str(model.score(X, df['label'])))

# Define a function to classify news
def classify_news(text):
    # Convert the text to a sparse matrix of term frequencies
    X_new = vectorizer.transform([text])

    # Make a prediction using the trained model
    prediction = model.predict(X_new)[0]

    # Return the prediction
    return prediction


# Open the text file in read mode
with open("article_the_onion.txt", "r") as f:
    article_text = f.read()
    print("The Onion article: " + classify_news(article_text))

with open("article_usa_today.txt", "r") as f:
    article_text = f.read()
    print("USA Today article: " + classify_news(article_text))
