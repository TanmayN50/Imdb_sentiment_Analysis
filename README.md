# IMDb Movie Review Sentiment Analysis

An end-to-end Deep Learning project that classifies English movie reviews as **Positive** or **Negative** using a Bidirectional LSTM model and a Streamlit web application.

## Project results

- Dataset: IMDb Dataset of 50K Movie Reviews
- Original rows: 50,000
- Duplicate rows removed during cleaning: 418
- Inputs: movie review text
- Target: sentiment (`negative` / `positive`)
- Model: Embedding + Bidirectional LSTM
- Reported test accuracy: **86.59%** (from the training notebook; results may vary if retrained)

## Features

- Data cleaning and exploratory data analysis (EDA)
- Text vectorization using TensorFlow/Keras `TextVectorization`
- Deep Learning classification with Bidirectional LSTM
- Evaluation with accuracy, confusion matrix, precision, recall, and F1-score
- Streamlit interface for testing a new review

## Folder structure

```text
IMDb_Sentiment_Analysis_Project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── imdb_sentiment_model.keras   # add your trained model file
└── imdb_vectorizer.pkl          # add your saved vectorizer file
```

The trained model and vectorizer are not included in this template ZIP. Download them from your Google Colab session and copy them into this folder before launching the app.

## Dataset

Use the IMDb Dataset of 50K Movie Reviews (CSV columns commonly named `review` and `sentiment`). Follow the dataset's license and terms when redistributing data. Do not upload the full dataset to GitHub unless its terms permit it.

## Setup

Recommended: Python 3.10 or 3.11 in a virtual environment. TensorFlow support depends on your operating system and Python version.

### 1. Create and activate a virtual environment

Windows:

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the trained artifacts

Download both files from Google Colab:

```python
from google.colab import files
files.download("imdb_sentiment_model.keras")
files.download("imdb_vectorizer.pkl")
```

Move the downloaded files beside `app.py`.

### 4. Run the web application

```bash
streamlit run app.py
```

Streamlit will show a local URL in the terminal, usually `http://localhost:8501`. Open it in your browser.

## How the model works

1. User enters an English movie review.
2. The saved vectorizer converts the text into integer sequences.
3. The trained model produces a score between 0 and 1.
4. The app uses a 0.5 threshold to show Positive or Negative.

## Evaluation

The reported test accuracy is 86.59%. Include the classification report and confusion matrix in your final project documentation after running the evaluation cells. Accuracy alone does not describe every kind of model error.

## Important notes

- The app expects the same saved vectorizer that was used when training the model.
- If loading the vectorizer fails, re-create and save the vectorizer using a supported Keras serialization approach or save the preprocessing layer as part of the model. Do not silently substitute a different vectorizer.
- The model can misclassify sarcasm, mixed sentiment, slang, and reviews that differ from the training data.
- Do not commit private credentials or large dataset files to GitHub.

## Resume project description

**IMDb Movie Review Sentiment Analysis | Deep Learning, TensorFlow/Keras, NLP, Streamlit**
- Built an end-to-end sentiment classification pipeline using 50,000 IMDb movie reviews.
- Cleaned and explored text data, removed 418 duplicate rows, and prepared training, validation, and test sets.
- Developed a Bidirectional LSTM model with an Embedding layer and text vectorization for binary sentiment classification.
- Evaluated the model using test accuracy and classification metrics; achieved 86.59% test accuracy in the recorded run.
- Created a Streamlit web app to accept user reviews and display predicted sentiment.

## Suggested GitHub repository description

End-to-end IMDb sentiment analysis using TensorFlow/Keras Bidirectional LSTM, NLP preprocessing, model evaluation, and a Streamlit web app.

## Suggested next improvements

- Compare LSTM with a baseline model such as TF-IDF + Logistic Regression.
- Tune vocabulary size, sequence length, embedding dimensions, and model architecture.
- Track precision, recall, F1-score, and confusion matrix.
- Add automated tests and deploy the app to a hosting service.
