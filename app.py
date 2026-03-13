from flask import Flask, render_template, request
import pandas as pd
import sqlite3

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from database import init_db

app = Flask(__name__)

init_db()

data = pd.read_csv("spam_dataset.csv")

X = data["text"]
y = data["label"]

vectorizer = CountVectorizer()

X_vector = vectorizer.fit_transform(X)

model = MultinomialNB()

model.fit(X_vector, y)


def save_message(message, prediction, probability):

    conn = sqlite3.connect("messages.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO messages (message, prediction, probability) VALUES (?, ?, ?)",
        (message, prediction, probability)
    )

    conn.commit()
    conn.close()


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    probability = None

    if request.method == "POST":

        message = request.form["message"]

        message_vector = vectorizer.transform([message])

        prediction = model.predict(message_vector)[0]

        prob = model.predict_proba(message_vector).max()

        probability = round(prob * 100, 2)

        result = prediction

        save_message(message, result, probability)

    return render_template("index.html", result=result, probability=probability)


@app.route("/history")
def history():

    conn = sqlite3.connect("messages.db")
    cursor = conn.cursor()

    cursor.execute("SELECT message, prediction, probability FROM messages")

    rows = cursor.fetchall()

    conn.close()

    return render_template("history.html", rows=rows)


if __name__ == "__main__":
    app.run(debug=True)