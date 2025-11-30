!pip install pandas scikit-learn matplotlib wordcloud nltk spacy textblob langdetect
!python -m spacy download en_core_web_sm

print("✅ All packages installed! Restart runtime and run the next cell.")

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
warnings.filterwarnings('ignore')


nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')


nlp = spacy.load('en_core_web_sm')

print("✅ All libraries loaded successfully!")

print("\n📊 Step 1: Creating Dataset...")

data = {
    'text': [
        'Hello, how are you today?',
        'I am doing great, thank you!',
        'Good morning, have a nice day!',
        'This is a test sentence.',
        'Welcome to our website.',
        
        'Bonjour, comment allez-vous?',
        'Je vais bien, merci beaucoup!',
        'Bonjour le monde!',
        'Comment ça va?',
        'Au revoir et merci!',
        
        'Hola, ¿cómo estás hoy?',
        'Estoy muy bien, gracias!',
        'Buenos días, que tengas un buen día!',
        'Esta es una oración de prueba.',
        'Bienvenido a nuestro sitio web.',
        
        'Hallo, wie geht es dir heute?',
        'Mir geht es gut, danke!',
        'Guten Morgen, einen schönen Tag!',
        'Das ist ein Test-Satz.',
        'Willkommen auf unserer Website.',
        
        'Ciao, come stai oggi?',
        'Sto bene, grazie mille!',
        'Buongiorno, buona giornata!',
        'Questa è una frase di prova.',
        'Benvenuto sul nostro sito web.'
    ],
    'language': [
        'English', 'English', 'English', 'English', 'English',
        'French', 'French', 'French', 'French', 'French',
        'Spanish', 'Spanish', 'Spanish', 'Spanish', 'Spanish',
        'German', 'German', 'German', 'German', 'German',
        'Italian', 'Italian', 'Italian', 'Italian', 'Italian'
    ]
}

df = pd.DataFrame(data)
print(f"Dataset created: {len(df)} samples across {df['language'].nunique()} languages")
print("\nSample data:")
print(df.head())

print("\n🔧 Step 2: Text Preprocessing...")

def preprocess_text(text):
    """Complete preprocessing pipeline using re, NLTK, spaCy"""
    # 1. Clean with regex (re)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    
    # 2. Lowercase
    text = text.lower().strip()
    
    # 3. NLTK: Tokenization + Stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # 4. spaCy: Lemmatization
    if tokens:
        doc = nlp(' '.join(tokens))
        lemmatized = [token.lemma_ for token in doc if not token.is_stop]
        return ' '.join(lemmatized)
    
    return text

df['cleaned_text'] = df['text'].apply(preprocess_text)
print("✅ Preprocessing complete!")
print("\nBefore vs After:")
print("Original:", df['text'].iloc[0])
print("Cleaned: ", df['cleaned_text'].iloc[0])

print("\n🔢 Step 3: Feature Extraction (TF-IDF)...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),  # Unigrams + Bigrams
    max_features=1000,
    lowercase=False  # Already lowercased
)

X = vectorizer.fit_transform(df['cleaned_text'])
y = df['language']

print(f"✅ TF-IDF Matrix: {X.shape[0]} samples × {X.shape[1]} features")

print("\n🤖 Step 4: Training Model...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = MultinomialNB()
model.fit(X_train, y_train)

print("\n📈 Step 5: Model Evaluation...")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {accuracy:.2%}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n🌩️ Step 6: Generating Word Clouds...")

plt.style.use('default')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, lang in enumerate(df['language'].unique()):
    text = ' '.join(df[df['language'] == lang]['cleaned_text'])
    wordcloud = WordCloud(
        width=400, height=300, 
        background_color='white',
        colormap='viridis'
    ).generate(text)
    
    axes[idx].imshow(wordcloud, interpolation='bilinear')
    axes[idx].set_title(f'{lang}\n({len(df[df["language"] == lang])} samples)', fontsize=14, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.suptitle('Word Clouds by Language', fontsize=16, y=1.02)
plt.show()

print("\n⚖️ Step 7: Baseline Comparison (langdetect)...")

print("TextBlob/langdetect Results:")
for i, text in enumerate(df['text'].head(10)):
    try:
        detected_lang = detect(text)
        true_lang = df['language'].iloc[i]
        status = "✅" if detected_lang.split('_')[0] == true_lang.lower()[:3] else "❌"
        print(f"{status} '{text[:30]}...' → True: {true_lang} | Detected: {detected_lang}")
    except:
        print(f"❌ '{text[:30]}...' → Detection failed")

print("\n🔮 Step 8: Test on New Sentences...")

new_texts = [
    "Hello world, how are you?",
    "Bonjour tout le monde!",
    "Hola mundo!",
    "Hallo Welt!",
    "Ciao mondo!"
]

new_texts_cleaned = [preprocess_text(text) for text in new_texts]
new_X = vectorizer.transform(new_texts_cleaned)
predictions = model.predict(new_X)

print("\n🆕 New Predictions:")
for text, pred in zip(new_texts, predictions):
    print(f"'{text}' → Predicted: {pred}")

print("\n🎉 PROJECT COMPLETE! All libraries used successfully!")
print("\n📚 Libraries Used:")
print("- Pandas: Data handling")
print("- re: Text cleaning")
print("- NLTK: Tokenization + stopwords")
print("- spaCy: Lemmatization")
print("- Scikit-learn: TF-IDF + Naive Bayes + Metrics")
print("- WordCloud: Visualization")
print("- langdetect: Baseline comparison")