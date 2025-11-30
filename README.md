# Multilingual Language Detector

This project is a simple end-to-end **language detection** pipeline built in Python. It detects the language of short text snippets (English, French, Spanish, German, Italian) using classic NLP preprocessing and a machine learning model.

## Features

- Cleans raw text with **regular expressions** (`re`)
- Tokenization and stopword removal using **NLTK**
- Lemmatization using **spaCy**
- Feature extraction with **TF-IDF** (Scikit-learn)
- Language classification with **Multinomial Naive Bayes**
- Evaluation with accuracy, classification report, and confusion matrix
- **WordCloud** visualizations for each language
- Baseline comparison using **langdetect**
- All logic contained in a single file: `main.py`

## Tech Stack / Libraries

- Python 3
- `pandas` – data handling
- `re` – text cleaning
- `nltk` – tokenization, stopwords
- `spacy` – lemmatization
- `scikit-learn` – TF-IDF, train/test split, Naive Bayes, metrics
- `wordcloud` – word cloud visualizations
- `matplotlib` – plotting
- `langdetect` – baseline language detection
- `textblob` – optional NLP utilities

## Setup

### 1. Clone the repository

git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>


### 2. Install dependencies

**In Google Colab (first cell):**

!pip install pandas scikit-learn matplotlib wordcloud nltk spacy textblob langdetect
!python -m spacy download en_core_web_sm

text


**Locally (terminal):**

pip install pandas scikit-learn matplotlib wordcloud nltk spacy textblob langdetect
python -m spacy download en_core_web_sm


### 3. Download NLTK data (if needed locally)

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


## How to Run

### In Google Colab

1. Upload `main.py` to Colab or open it directly from GitHub.
2. Install the dependencies as shown above.
3. Run:

%run main.py


The script will:

- Create a small multilingual dataset
- Preprocess the text
- Train the Naive Bayes classifier
- Print evaluation metrics
- Show word cloud visualizations
- Test on new example sentences

### On Local Machine

python main.py


## Project Structure

.
├── main.py # Full language detection pipeline
└── README.md # Project documentation


## Explanation Highlights

You can explain this project in terms of:

- **Problem**: Detect the language of short text inputs.
- **Pipeline**: Cleaning → tokenization → lemmatization → TF-IDF → Naive Bayes → evaluation.
- **Libraries**: Role of NLTK, spaCy, pandas, Scikit-learn, TextBlob, WordCloud, `re`, and `langdetect`.
- **Results**: Accuracy, confusion matrix, and language-specific word clouds.

This repository demonstrates a clear, beginner-friendly NLP workflow using multiple text analytics libraries in Python.
