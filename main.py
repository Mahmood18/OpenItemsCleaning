import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



# Load the trained model, tokenizer, and encoder
model = tf.keras.models.load_model('models/bidirectional_lstm_product_classification_model.h5')
tokenizer = joblib.load('models/tokenizer.pkl')
encoder = joblib.load('models/label_encoder.pkl')

# Define the max sequence length as used during training
max_sequence_len = 100  # Adjust based on your model's training

# Define stopwords
stop_words = set(stopwords.words('english'))

# Define measurement types
measurement_types = ['ml', 'unit', 'gm', 'units', 'pcs', 'kg', 'litre', 'mg', 
                     'ltr', 'oz', 'gb', 'litr', 'piece', 'box']

# Preprocessing function to clean the product name
def preprocess_text(text):
    text = text.lower()
    # Remove anything that is not an alphabetic character (keep only a-z and A-Z)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Function to check if a product name should be classified as 'MISCELLANEOUS'
def is_miscellaneous(product_name):
    product_name = product_name.strip().lower()
    
    tokens = word_tokenize(product_name)
    
    words = []
    measurements = []
    
    for token in tokens:
        if any(measurement in token for measurement in measurement_types) or re.match(r'\d+(\.\d+)?', token):
            measurements.append(token)
        else:
            words.append(token)
    
    if len(words) == 0 and len(measurements) > 0:
        return True
    
    if all(token.isdigit() or any(measurement in token for measurement in measurement_types) for token in tokens):
        return True
    
    return False

# Function to predict categories for a dataframe
def predict_categories(df):
    df['ProductName'] = df['ProductName'].apply(preprocess_text)
    sequences = tokenizer.texts_to_sequences(df['ProductName'])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_len)
    predictions = model.predict(padded_sequences)
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_categories = encoder.inverse_transform(predicted_labels)
    df['PredictedSubCategory'] = predicted_categories
    return df

# Streamlit app layout
st.title("OpenItem Subcategory Predictor")
st.write("Upload a CSV file with a 'ProductName' column.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'ProductName' in df.columns:
        st.write("Data preview:")
        st.dataframe(df.head())

        # Automatically predict categories upon file upload
        result_df = predict_categories(df)
        st.write("Predictions completed. Preview:")
        st.dataframe(result_df)

        # Download the result
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV with Predictions",
            data=csv,
            file_name='predicted_subcategories.csv',
            mime='text/csv',
        )
    else:
        st.error("The uploaded CSV does not contain a 'ProductName' column.")
