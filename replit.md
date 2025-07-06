# Fake News Detection System

## Overview

This is a Streamlit-based web application that uses machine learning to detect fake news articles and headlines. The system analyzes text input and provides predictions on whether the content is likely to be real or fake news. The application is built with Python and uses scikit-learn for machine learning capabilities.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web interface
- **Layout**: Wide layout with collapsed sidebar for clean user experience
- **State Management**: Streamlit session state for model caching and loading status
- **Caching**: `@st.cache_resource` decorator for efficient model loading

### Backend Architecture
- **ML Framework**: scikit-learn for machine learning models
- **Text Processing**: Custom preprocessing pipeline using regex and string operations
- **Model Storage**: Pickle-based serialization for model persistence
- **Fallback System**: Placeholder model creation when trained models are unavailable

### Data Processing Pipeline
- **Text Preprocessing**: Multi-stage cleaning including URL removal, case normalization, and punctuation handling
- **Vectorization**: TF-IDF or Count vectorization for text feature extraction
- **Model Inference**: Real-time prediction using loaded ML models

## Key Components

### 1. Main Application (`app.py`)
- **Purpose**: Primary Streamlit interface and application orchestration
- **Responsibilities**: 
  - UI rendering and user interaction
  - Model loading and caching
  - Session state management
  - Error handling and user feedback

### 2. Model Utilities (`model_utils.py`)
- **Purpose**: Model loading and management functionality
- **Responsibilities**:
  - Loading trained models and vectorizers from disk
  - Creating placeholder models for demonstration
  - Error handling for missing model files
  - Supporting multiple model formats (pickle, joblib)

### 3. Text Preprocessing (`preprocessing.py`)
- **Purpose**: Text cleaning and normalization pipeline
- **Responsibilities**:
  - URL and email removal
  - Social media content cleaning (mentions, hashtags)
  - HTML tag removal
  - Whitespace and punctuation normalization
  - Case standardization

## Data Flow

1. **User Input**: User pastes news text into Streamlit interface
2. **Preprocessing**: Text goes through cleaning pipeline (`preprocess_text()`)
3. **Vectorization**: Cleaned text is converted to numerical features using loaded vectorizer
4. **Prediction**: Vectorized features are fed to ML model for classification
5. **Results Display**: Prediction results are formatted and displayed to user

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and utilities
- **pickle/joblib**: Model serialization and deserialization

### Text Processing
- **re**: Regular expression operations for text cleaning
- **string**: String manipulation utilities

## Deployment Strategy

### Current Setup
- **Platform**: Designed for Replit deployment
- **Model Storage**: Local file system using pickle format
- **Scalability**: Single-instance application with in-memory model caching

### Model Integration
- **Training Environment**: Models trained in Google Colab
- **Transfer Method**: Manual upload of model files (`model_placeholder.pkl`, `vectorizer.pkl`)
- **Fallback Strategy**: Placeholder model creation when trained models unavailable

### Performance Considerations
- **Caching**: Model loaded once and cached in session state
- **Memory Management**: Efficient vectorizer and model storage
- **Response Time**: Real-time prediction with minimal latency

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes

- July 06, 2025: Added historical prediction tracking and analytics dashboard
  - Created PredictionTracker class for data persistence
  - Added analytics dashboard with charts and metrics
  - Implemented trend analysis and confidence distribution
  - Added session tracking and prediction history management

## Changelog

- July 06, 2025: Initial setup with fake news detection system
- July 06, 2025: Added analytics and prediction tracking features