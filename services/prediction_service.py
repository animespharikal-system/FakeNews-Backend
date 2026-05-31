import re
from pathlib import Path

import joblib

from models.prediction import PredictionResponse


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models_ml"
MODEL_PATH = MODEL_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class PredictionService:
    def __init__(self) -> None:
        self.model = self._load_artifact(MODEL_PATH)
        self.vectorizer = self._load_artifact(VECTORIZER_PATH)

    @staticmethod
    def _load_artifact(path: Path) -> object:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing ML artifact: {path}. Run 'python scripts/train_model.py' first."
            )

        return joblib.load(path)

    def predict(self, text: str) -> PredictionResponse:
        processed_text = preprocess_text(text)
        features = self.vectorizer.transform([processed_text])
        prediction = int(self.model.predict(features)[0])
        if hasattr(self.model, "predict_proba"):
           probabilities = self.model.predict_proba(features)[0]
           class_index = list(self.model.classes_).index(prediction)
           confidence = float(probabilities[class_index] * 100)
        else:
           confidence = 100.0
        label = "Fake" if prediction == 0 else "Real"
        return PredictionResponse(
           prediction=label,
           confidence=round(confidence, 2),
        )