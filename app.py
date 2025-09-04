
from flask import Flask, render_template, request
from transformers import pipeline

# Load Hugging Face pipeline (sentiment analysis)
classifier = pipeline("sentiment-analysis")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        text = request.form["user_input"]
        result = classifier(text)[0]  # Example: {'label': 'POSITIVE', 'score': 0.99}
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)