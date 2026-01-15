import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from langdetect import detect
import warnings
import joblib
import os
import sys

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
MODEL_PATH = 'language_model.joblib'
VECTORIZER_PATH = 'vectorizer.joblib'

# --- SETUP ---
def setup_nltk():
    """Download necessary NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')

# Load SpaCy model globally for efficiency
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    sys.exit(1)


# --- DATA GENERATION ---
def create_dataset():
    """Create a synthetic dataset for training."""
    print("📊 Creating Dataset...")
    data = {
        'text': [
            'Hello, how are you today?', 'I am doing great, thank you!', 'Good morning, have a nice day!', 
            'This is a test sentence.', 'Welcome to our website.',
            'Bonjour, comment allez-vous?', 'Je vais bien, merci beaucoup!', 'Bonjour le monde!', 
            'Comment ça va?', 'Au revoir et merci!',
            'Hola, ¿cómo estás hoy?', 'Estoy muy bien, gracias!', 'Buenos días, que tengas un buen día!', 
            'Esta es una oración de prueba.', 'Bienvenido a nuestro sitio web.',
            'Hallo, wie geht es dir heute?', 'Mir geht es gut, danke!', 'Guten Morgen, einen schönen Tag!', 
            'Das ist ein Test-Satz.', 'Willkommen auf unserer Website.',
            'Ciao, come stai oggi?', 'Sto bene, grazie mille!', 'Buongiorno, buona giornata!', 
            'Questa è una frase di prova.', 'Benvenuto sul nostro sito web.'
        ],
        'language': [
            'English', 'English', 'English', 'English', 'English',
            'French', 'French', 'French', 'French', 'French',
            'Spanish', 'Spanish', 'Spanish', 'Spanish', 'Spanish',
            'German', 'German', 'German', 'German', 'German',
            'Italian', 'Italian', 'Italian', 'Italian', 'Italian'
        ]
    }
    return pd.DataFrame(data)

# --- PREPROCESSING ---
def preprocess_text(text):
    """Clean and lemmatize text."""
    # 1. Clean with regex
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # 2. Lowercase
    text = text.lower().strip()
    
    # 3. NLTK Tokenization
    stop_words = set(stopwords.words('english')) # Note: Using English stopwords for all is a simplification
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # 4. SpaCy Lemmatization
    if tokens:
        doc = nlp(' '.join(tokens))
        lemmatized = [token.lemma_ for token in doc if not token.is_stop]
        return ' '.join(lemmatized)
    
    return text

# --- TRAINING ---
def train_model(df):
    """Train the model and return vectorizer and model."""
    print("\n🔧 Preprocessing and Feature Extraction...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000, lowercase=False)
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['language']
    
    print(f"✅ TF-IDF Matrix: {X.shape[0]} samples × {X.shape[1]} features")
    
    print("\n🤖 Training Model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {accuracy:.2%}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return vectorizer, model, df

# --- PERSISTENCE ---
def save_artifacts(vectorizer, model):
    """Save model and vectorizer to disk."""
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    print(f"\n💾 Model saved to {MODEL_PATH} and {VECTORIZER_PATH}")

def load_artifacts():
    """Load model and vectorizer from disk."""
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)
        return vectorizer, model
    return None, None

# --- VISUALIZATION ---
def plot_wordclouds(df):
    """Generate and show word clouds."""
    print("\n🌩️ Generating Word Clouds...")
    plt.style.use('default')
    unique_langs = df['language'].unique()
    rows = (len(unique_langs) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5*rows))
    axes = axes.ravel()
    
    for idx, lang in enumerate(unique_langs):
        text = ' '.join(df[df['language'] == lang]['cleaned_text'])
        if not text: continue
        wordcloud = WordCloud(width=400, height=300, background_color='white', colormap='viridis').generate(text)
        axes[idx].imshow(wordcloud, interpolation='bilinear')
        axes[idx].set_title(f'{lang}', fontsize=14, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide unused axes
    for i in range(len(unique_langs), len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()

# --- PREDICTION ---
def predict_language(text, vectorizer, model):
    """Predict language for a single string."""
    cleaned = preprocess_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    return pred, prob

# --- MAIN ---
def main():
    setup_nltk()

    # CLI Argument to decide mode? For now, we'll ask user or check if model exists.
    # But for this simple script, we'll train if model doesn't exist, then loop.
    
    vectorizer, model = load_artifacts()
    
    if vectorizer is None or model is None:
        print("Model not found. Training new model...")
        df = create_dataset()
        vectorizer, model, df_processed = train_model(df)
        save_artifacts(vectorizer, model)
        plot_wordclouds(df_processed)
    else:
        print("✅ Model loaded successfully.")

    print("\n🔮 Interactive Mode (type 'exit' to quit)")
    while True:
        user_input = input("\nEnter text: ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
        
        pred_lang, confidence = predict_language(user_input, vectorizer, model)
        
        # Comparison with langdetect
        try:
            baseline_lang = detect(user_input)
        except:
            baseline_lang = "unknown"
            
        print(f"👉 Predicted: {pred_lang} ({confidence:.2%} confidence)")
        print(f"⚖️  Baseline (langdetect): {baseline_lang}")

if __name__ == "__main__":
    main()