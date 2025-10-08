# Spam & Phishing Detection System

A machine learning-powered web application that detects spam messages, phishing attempts, and malicious URLs using multiple classification models. Built with Flask and scikit-learn, this project demonstrates expertise in web development, machine learning, and cybersecurity applications.

## 🚀 Live Demo

🌐 **[Try the live application here](https://spam-phishing-detector.onrender.com/)**

## ✨ Features

- **Multi-Model Detection**: Three specialized models for different threat types
- **Real-time Analysis**: Instant classification with confidence scores
- **Web Interface**: Clean, responsive UI built with Bootstrap
- **RESTful API**: JSON-based API for integration with other applications
- **Scalable Architecture**: Modular design for easy feature expansion
- **Security Focus**: Addresses real-world cybersecurity challenges

## 🤖 Machine Learning Models

### 1. Spam Detection Model
- **Algorithm**: Multinomial Naive Bayes
- **Dataset**: 5,570 SMS messages
- **Features**: Text-based (CountVectorizer)
- **Accuracy**: Optimized for SMS spam detection

### 2. Phishing Message Detection
- **Algorithm**: Multinomial Naive Bayes
- **Dataset**: 10,000+ URL feature records
- **Features**: URL characteristics (dots, subdomains, path levels)
- **Purpose**: Detects phishing attempts in messages

### 3. Phishing URL Detection
- **Algorithm**: Multinomial Naive Bayes
- **Dataset**: 11,430+ URLs with 90+ features
- **Features**: URL length, domain analysis, security indicators
- **Purpose**: Direct URL threat assessment

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/spam-phishing-detector.git
   cd spam-phishing-detector
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python SpamDetection.py
   ```

5. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

## 📖 Usage

### Web Interface
1. Enter a message or URL in the text area
2. Select the detection type:
   - **Spam**: For SMS/text message analysis
   - **Phishing**: For phishing message detection
   - **Phishing URL**: For direct URL analysis
3. Click "Check" to get instant results
4. View the classification result and confidence score

## 💻 Technologies Used

### Backend
- **Flask 3.0.3**: Web framework
- **Pandas 2.2.3**: Data manipulation
- **Scikit-learn 1.5.2**: Machine learning
- **NumPy 2.0.2**: Numerical computing

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Custom styling
- **Bootstrap 4.5.2**: Responsive design
- **JavaScript**: Interactive functionality

### Development Tools
- **Git**: Version control
- **Pip**: Package management
- **Virtual Environment**: Dependency isolation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Kevin Dinh**
- Email: [kkd49@drexel.edu](mailto:kkd49@drexel.edu)
- LinkedIn: [kevinkdinh](https://linkedin.com/in/kevinkdinh)