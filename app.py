from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('logistic_regression_NLPreviews.joblib')
vectorizer = joblib.load('vectorizer_reviews.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the review from the POST request
    review = request.form['review']

    # Vectorize the review
    review_vector = vectorizer.transform([review])

    # Make a prediction
    prediction = model.predict(review_vector)

    # Convert the prediction to a JSON response
    output = 'positive' if prediction[0] == 1 else 'negative'
    
    return render_template('index.html', prediction_text='The Restaurant review was {}'.format(output))

if __name__ == '__main__':
    app.run(port=5000, debug=True)

