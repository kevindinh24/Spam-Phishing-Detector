from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Global variables for models and vectorizers
models = {
    'spam': None,
    'phishing': None,
    'phishing_url': None
}
vectorizers = {
    'spam': None,
    'phishing': None,
    'phishing_url': None
}

# Function to load spam dataset
def load_spam_data(file=None):
    if file is not None:
        data = pd.read_csv(file)
    else:
        data = pd.read_csv("/Users/kevindinh/Desktop/Spam/spam.csv")  # Your specific path
       
    data.drop_duplicates(inplace=True)
    data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])
    return data

# Function to load phishing message dataset
def load_phishing_data(file=None):
    if file is not None:
        data = pd.read_csv(file)
    else:
        data = pd.read_csv("/Users/kevindinh/Desktop/Spam/phishing.csv")  # Your specific path
       
    data['Category'] = data['CLASS_LABEL'].replace([0, 1], ['Not Phishing', 'Phishing'])
    data['Message'] = data.apply(lambda row: f"{row['NumDots']} {row['SubdomainLevel']} {row['PathLevel']} {row['UrlLength']}", axis=1)
    return data

# Function to load phishing URL dataset
def load_phishing_url_data(file=None):
    if file is not None:
        data = pd.read_csv(file)
    else:
        data = pd.read_csv("/Users/kevindinh/Desktop/Spam/phishing_url.csv")  # Your specific path
       
    data['Category'] = data['status'].replace(['legitimate', 'phishing'], ['Not Phishing', 'Phishing'])
    data['Message'] = data['url']  # Use URLs directly as the message for vectorization
    return data

# Preprocessing and model training
def train_model(data):
    mess = data['Message']
    cat = data['Category']

    # Split 80% train, 20% test
    (mess_train, mess_test, cat_train, cat_test) = train_test_split(mess, cat, test_size=0.2)

    # Vectorize the text data
    cv = CountVectorizer(stop_words='english')
    features = cv.fit_transform(mess_train)

    # Create and train the model
    model = MultinomialNB()
    model.fit(features, cat_train)

    return model, cv

# Prediction function
def predict(message, model, cv):
    input_message = cv.transform([message])  # Vectorize the input message
    proba = model.predict_proba(input_message)  # Get probability estimates
    result = model.predict(input_message)  # Predict the result
    return result[0], proba[0]  # Return prediction and probabilities

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/LICENSE')
def license_file():
    return send_from_directory('.', 'LICENSE')

# Route for handling the form submission
@app.route('/predict', methods=['POST'])
def predict_route():
    message = request.form['message']
    detection_type = request.form['detection_type']
    
    if detection_type == 'Spam':
        data = load_spam_data()
    elif detection_type == 'Phishing':
        data = load_phishing_data()
    else:  # Phishing URL
        data = load_phishing_url_data()
    
    # Train the model using the selected dataset
    model, cv = train_model(data)
    output, proba = predict(message, model, cv)
    confidence = proba[1]  # Probability of phishing/spam
    
    return jsonify({
        'result': output,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)

