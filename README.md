# Fake News Detection Backend

FastAPI backend for a Fake News Detection application. The current prediction endpoint uses mock data and is ready to be replaced with a real model service.

## Project Structure

```text
.
├── main.py
├── requirements.txt
├── .env.example
├── models
│   └── prediction.py
├── routes
│   ├── predict.py
│   └── status.py
├── services
│   └── prediction_service.py
└── utils
    └── config.py
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create your environment file:

```bash
copy .env.example .env
```

4. Run the API:

```bash
uvicorn main:app --reload
```

The server will start at:

```text
http://127.0.0.1:8000
```

## Endpoints

### GET /

Returns API status.

Example response:

```json
{
  "status": "ok",
  "message": "Fake News Detection API is running"
}
```

## Training the ML Classifier

Train or refresh the TF-IDF + Logistic Regression classifier with:

```bash
python scripts/train_model.py
```

By default, the script looks for:

```text
data/Fake.csv
data/True.csv
```

Expected CSV columns include `text`. Rows from `Fake.csv` are labeled `0`, and rows from `True.csv` are labeled `1`. Trained artifacts are saved to:

```text
models_ml/fake_news_model.pkl
models_ml/tfidf_vectorizer.pkl
```

### POST /predict

Request:

```json
{
  "text": "news article text"
}
```

Response:

```json
{
  "prediction": 0,
  "confidence": 95.2
}
```

## API Docs

FastAPI docs are available after startup:

```text
http://127.0.0.1:8000/docs
```
