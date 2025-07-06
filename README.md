# Fake News Detection System

A Streamlit-based web application that uses machine learning to detect fake news articles and headlines with comprehensive analytics tracking.

## Features

- **Real-time News Analysis**: Analyze news articles and headlines for authenticity
- **Machine Learning Integration**: Support for custom models trained in Google Colab
- **Historical Tracking**: Automatic prediction history with data persistence
- **Analytics Dashboard**: Comprehensive analytics with charts and trends
- **Confidence Scoring**: Detailed confidence levels for each prediction
- **Text Preprocessing**: Advanced text cleaning and normalization

## Project Structure

```
fake-news-detection/
├── app.py                    # Main Streamlit application
├── model_utils.py           # Model loading and prediction functions
├── preprocessing.py         # Text preprocessing utilities
├── analytics.py             # Prediction tracking and analytics
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── replit.md              # Project documentation and preferences
```

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

### Basic Usage
1. Open the application in your browser
2. Navigate to the "Analyze News" tab
3. Paste news text in the input area
4. Click "Analyze News" to get predictions
5. View results with confidence scores

### Analytics
1. Navigate to the "Analytics" tab
2. View prediction history and trends
3. Analyze confidence distributions
4. Monitor daily prediction patterns

### Using Your Own Model

To use a model trained in Google Colab:

1. In your Colab notebook, save your model:
   ```python
   import pickle
   
   # Save your trained model
   with open('trained_model.pkl', 'wb') as f:
       pickle.dump(your_model, f, protocol=2)
   
   # Save your vectorizer
   with open('vectorizer.pkl', 'wb') as f:
       pickle.dump(your_vectorizer, f, protocol=2)
   
   # Download files
   from google.colab import files
   files.download('trained_model.pkl')
   files.download('vectorizer.pkl')
   ```

2. Upload both files to the project directory
3. Restart the application

## Technical Details

### Model Requirements
- Compatible with scikit-learn models (Logistic Regression, Random Forest, etc.)
- Supports TF-IDF and CountVectorizer
- Expects binary classification (0 = Real, 1 = Fake)

### Data Storage
- Prediction history stored in `prediction_history.json`
- Automatic cleanup (keeps last 1000 predictions)
- Session tracking for analytics

### Text Preprocessing
- URL and email removal
- HTML tag cleaning
- Social media content normalization
- Punctuation and whitespace handling

## Configuration

Edit `.streamlit/config.toml` to customize:
- Server settings
- Theme preferences
- Port configuration

## Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- pickle/joblib

## License

This project is open source and available under the MIT License.

## Support

For issues and questions, please refer to the project documentation or create an issue in the repository.