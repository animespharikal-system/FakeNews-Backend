import json
import logging
from pathlib import Path

import joblib

from models.prediction import PredictionResponse
from utils.text_preprocessing import preprocess_text


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models_ml"
MODEL_PATH = MODEL_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
FALLBACK_LABEL_NAMES = {
    0: "Fake",
    1: "Real",
}

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self) -> None:
        self.model = self._load_artifact(MODEL_PATH)
        self.vectorizer = self._load_artifact(VECTORIZER_PATH)
        self.metadata = self._load_metadata(METADATA_PATH)
        self.label_names = self._load_label_names()
        logger.info(
            "Loaded ML artifacts: model=%s, vectorizer=%s, classes=%s, labels=%s",
            MODEL_PATH,
            VECTORIZER_PATH,
            getattr(self.model, "classes_", None),
            self.label_names,
        )

    @staticmethod
    def _load_artifact(path: Path) -> object:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing ML artifact: {path}. Run 'python scripts/train_model.py' first."
            )

        return joblib.load(path)

    @staticmethod
    def _load_metadata(path: Path) -> dict[str, object]:
        if not path.exists():
            logger.warning(
                "Missing model metadata: %s. Falling back to default label mapping.",
                path,
            )
            return {}

        with path.open("r", encoding="utf-8") as metadata_file:
            return json.load(metadata_file)

    def _load_label_names(self) -> dict[int, str]:
        raw_label_names = self.metadata.get("label_names")
        if not isinstance(raw_label_names, dict):
            return FALLBACK_LABEL_NAMES

        return {int(label): str(name) for label, name in raw_label_names.items()}

    def predict(self, text: str) -> PredictionResponse:
        processed_text = preprocess_text(text)
        features = self.vectorizer.transform([processed_text])
        prediction = int(self.model.predict(features)[0])
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
            class_probabilities = {
                int(class_label): float(probabilities[index] * 100)
                for index, class_label in enumerate(self.model.classes_)
            }
            confidence = class_probabilities[prediction]
        else:
            class_probabilities = {prediction: 100.0}
            confidence = 100.0
        label = self.label_names.get(prediction, str(prediction))
        logger.debug(
            "Prediction completed: raw_chars=%d, processed_chars=%d, nnz=%d, "
            "prediction=%s, confidence=%.2f, probabilities=%s",
            len(text),
            len(processed_text),
            features.nnz,
            label,
            confidence,
            {
                self.label_names.get(label_id, str(label_id)): round(probability, 2)
                for label_id, probability in class_probabilities.items()
            },
        )
        return PredictionResponse(
            prediction=label,
            confidence=round(confidence, 2),
        )
