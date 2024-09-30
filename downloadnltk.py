import nltk
# Specify the path where you want to store the NLTK data
nltk_data_path = 'C:\\Users\\IT\\AppData\\Local\\Programs\\Python\\Python312\\share\\nltk_data'
nltk.data.path.append(nltk_data_path)
# Download 'punkt' tokenizer models
nltk.download('punkt', download_dir=nltk_data_path)
