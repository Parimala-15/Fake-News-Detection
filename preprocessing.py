import re
import string
import pandas as pd

def preprocess_text(text):
    """
    Preprocess text for fake news detection model.
    
    This function applies common text preprocessing steps that are typically
    used in NLP models for fake news detection. The preprocessing should
    match the steps used during model training.
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove mentions and hashtags (common in social media text)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove numbers (optional - depends on training approach)
    # text = re.sub(r'\d+', '', text)
    
    # Remove punctuation (optional - some models benefit from keeping punctuation)
    # You may want to comment this out if your model was trained with punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra spaces again after punctuation removal
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def advanced_preprocess_text(text):
    """
    Advanced preprocessing with additional steps.
    
    Use this function if your model was trained with more sophisticated
    preprocessing steps.
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Preprocessed text
    """
    # Start with basic preprocessing
    text = preprocess_text(text)
    
    # Remove repeated characters (e.g., "sooooo" -> "so")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Remove single characters (except 'a' and 'i')
    text = re.sub(r'\b[b-hj-z]\b', '', text, flags=re.IGNORECASE)
    
    # Handle contractions (basic approach)
    contractions = {
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am"
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_features(text):
    """
    Extract additional features from text that might be useful for fake news detection.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary of extracted features
    """
    features = {}
    
    # Basic text statistics
    features['char_count'] = len(text)
    features['word_count'] = len(text.split())
    features['sentence_count'] = len(re.split(r'[.!?]+', text))
    features['avg_word_length'] = sum(len(word) for word in text.split()) / max(len(text.split()), 1)
    
    # Punctuation features
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    
    # Emotional indicators
    emotional_words = ['amazing', 'shocking', 'unbelievable', 'incredible', 'urgent', 'breaking']
    features['emotional_word_count'] = sum(1 for word in emotional_words if word in text.lower())
    
    # Clickbait indicators
    clickbait_phrases = ['you won\'t believe', 'click here', 'what happens next', 'doctors hate']
    features['clickbait_indicators'] = sum(1 for phrase in clickbait_phrases if phrase in text.lower())
    
    return features

def validate_input(text):
    """
    Validate user input text.
    
    Args:
        text (str): User input text
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not text or not isinstance(text, str):
        return False, "Please enter some text to analyze."
    
    # Remove whitespace for length check
    clean_text = text.strip()
    
    if len(clean_text) < 10:
        return False, "Please enter at least 10 characters for meaningful analysis."
    
    if len(clean_text) > 10000:
        return False, "Text is too long. Please limit to 10,000 characters."
    
    # Check if text contains meaningful content (not just special characters)
    meaningful_chars = re.sub(r'[^\w\s]', '', clean_text)
    if len(meaningful_chars) < 5:
        return False, "Please enter text with meaningful content."
    
    return True, "Valid input"
