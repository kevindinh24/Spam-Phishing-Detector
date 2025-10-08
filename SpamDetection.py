from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import os

app = Flask(__name__)

# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    try:
        if file is not None:
            data = pd.read_csv(file)
        else:
            data = pd.read_csv(os.path.join(BASE_DIR, "spam.csv"))
           
        data.drop_duplicates(inplace=True)
        data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])
        return data
    except FileNotFoundError:
        raise Exception("Spam dataset not found. Please ensure spam.csv is in the project directory.")
    except Exception as e:
        raise Exception(f"Error loading spam data: {str(e)}")

# Function to load phishing message dataset
def load_phishing_data(file=None):
    try:
        if file is not None:
            data = pd.read_csv(file)
        else:
            data = pd.read_csv(os.path.join(BASE_DIR, "phishing.csv"))
           
        data['Category'] = data['CLASS_LABEL'].replace([0, 1], ['Not Phishing', 'Phishing'])
        data['Message'] = data.apply(lambda row: f"{row['NumDots']} {row['SubdomainLevel']} {row['PathLevel']} {row['UrlLength']}", axis=1)
        return data
    except FileNotFoundError:
        raise Exception("Phishing dataset not found. Please ensure phishing.csv is in the project directory.")
    except Exception as e:
        raise Exception(f"Error loading phishing data: {str(e)}")

# Function to load phishing URL dataset
def load_phishing_url_data(file=None):
    try:
        if file is not None:
            data = pd.read_csv(file)
        else:
            data = pd.read_csv(os.path.join(BASE_DIR, "phishing_url.csv"))
           
        data['Category'] = data['status'].replace(['legitimate', 'phishing'], ['Not Phishing', 'Phishing'])
        data['Message'] = data['url']  # Use URLs directly as the message for vectorization
        return data
    except FileNotFoundError:
        raise Exception("Phishing URL dataset not found. Please ensure phishing_url.csv is in the project directory.")
    except Exception as e:
        raise Exception(f"Error loading phishing URL data: {str(e)}")

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
    try:
        message = request.form['message']
        detection_type = request.form['detection_type']
        
        if not message.strip():
            return jsonify({'error': 'Please enter a message to analyze'}), 400
        
        if detection_type == 'Spam':
            data = load_spam_data()
        elif detection_type == 'Phishing':
            data = load_phishing_data()
        elif detection_type == 'Phishing URL':
            data = load_phishing_url_data()
        else:
            return jsonify({'error': 'Invalid detection type selected'}), 400
        
        # Train the model using the selected dataset
        model, cv = train_model(data)
        output, proba = predict(message, model, cv)
        confidence = proba[1]  # Probability of phishing/spam
        
        return jsonify({
            'result': output,
            'confidence': round(confidence, 3)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Run in production mode when deployed
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)

