import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utils.text_preprocessing import (
    PREPROCESSING_VERSION,
    SOURCE_MARKERS,
    preprocess_text,
)


DEFAULT_FAKE_PATH = ROOT_DIR / "data" / "Fake.csv"
DEFAULT_TRUE_PATH = ROOT_DIR / "data" / "True.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "models_ml"
MODEL_FILENAME = "fake_news_model.pkl"
VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"
METADATA_FILENAME = "model_metadata.json"
FAKE_LABEL = 0
TRUE_LABEL = 1
LABEL_NAMES = {
    FAKE_LABEL: "Fake",
    TRUE_LABEL: "Real",
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArticleRecord:
    text: str
    label: int
    source_file: str


def load_labeled_articles(
    csv_path: Path,
    label: int,
    text_columns: list[str],
) -> list[ArticleRecord]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    records: list[ArticleRecord] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {csv_path}")
        missing_columns = [
            column for column in text_columns if column not in reader.fieldnames
        ]
        if missing_columns:
            available_columns = ", ".join(reader.fieldnames)
            raise ValueError(
                f"Missing text column(s) {missing_columns!r} in {csv_path}. "
                f"Available columns: {available_columns}"
            )

        for row in reader:
            text = " ".join(
                (row.get(column) or "").strip() for column in text_columns
            ).strip()
            if text:
                records.append(
                    ArticleRecord(text=text, label=label, source_file=csv_path.name)
                )

    if not records:
        raise ValueError(f"No usable article text found in {csv_path}")

    return records


def load_training_records(
    fake_path: Path,
    true_path: Path,
    text_columns: list[str],
) -> list[ArticleRecord]:
    fake_records = load_labeled_articles(fake_path, FAKE_LABEL, text_columns)
    true_records = load_labeled_articles(true_path, TRUE_LABEL, text_columns)
    logger.info(
        "Loaded dataset: %s=%d (%s), %s=%d (%s), text_columns=%s",
        LABEL_NAMES[FAKE_LABEL],
        len(fake_records),
        fake_path,
        LABEL_NAMES[TRUE_LABEL],
        len(true_records),
        true_path,
        text_columns,
    )
    return fake_records + true_records


def split_records(
    records: list[ArticleRecord],
    test_size: float,
    random_state: int,
) -> tuple[list[str], list[str], list[int], list[int]]:
    texts = [record.text for record in records]
    labels = [record.label for record in records]

    if len(set(labels)) != 2:
        raise ValueError("Training data must contain both fake and true articles.")

    return train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        max_features=250_000,
        sublinear_tf=True,
    )


def build_model() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )


def calculate_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def print_metrics(metrics: dict[str, float]) -> None:
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")


def train_and_evaluate(
    records: list[ArticleRecord],
    test_size: float,
    random_state: int,
) -> tuple[LogisticRegression, TfidfVectorizer, dict[str, float]]:
    x_train, x_test, y_train, y_test = split_records(records, test_size, random_state)
    x_train = [preprocess_text(text) for text in x_train]
    x_test = [preprocess_text(text) for text in x_test]

    vectorizer = build_vectorizer()
    x_train_features = vectorizer.fit_transform(x_train)
    x_test_features = vectorizer.transform(x_test)
    logger.info(
        "Evaluation vectorizer fitted: train_rows=%d, test_rows=%d, vocab_size=%d",
        x_train_features.shape[0],
        x_test_features.shape[0],
        len(vectorizer.vocabulary_),
    )

    model = build_model()
    model.fit(x_train_features, y_train)
    y_pred = model.predict(x_test_features)
    metrics = calculate_metrics(y_test, y_pred)

    print_metrics(metrics)
    logger.info("Evaluation metrics: %s", metrics)

    all_texts = [preprocess_text(record.text) for record in records]
    all_labels = [record.label for record in records]
    final_vectorizer = build_vectorizer()
    all_features = final_vectorizer.fit_transform(all_texts)
    final_model = build_model()
    final_model.fit(all_features, all_labels)
    logger.info(
        "Final model fitted: rows=%d, vocab_size=%d, classes=%s",
        all_features.shape[0],
        len(final_vectorizer.vocabulary_),
        final_model.classes_.tolist(),
    )

    return final_model, final_vectorizer, metrics


def save_artifacts(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    output_dir: Path,
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / MODEL_FILENAME
    vectorizer_path = output_dir / VECTORIZER_FILENAME
    metadata_path = output_dir / METADATA_FILENAME

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model to: {model_path}")
    print(f"Saved vectorizer to: {vectorizer_path}")
    print(f"Saved metadata to: {metadata_path}")
    logger.info("Saved artifacts to %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a TF-IDF + Logistic Regression fake news classifier."
    )
    parser.add_argument(
        "--fake-data",
        type=Path,
        default=DEFAULT_FAKE_PATH,
        help="Path to Fake.csv. Rows from this file are labeled 0.",
    )
    parser.add_argument(
        "--true-data",
        type=Path,
        default=DEFAULT_TRUE_PATH,
        help="Path to True.csv. Rows from this file are labeled 1.",
    )
    parser.add_argument(
        "--text-columns",
        nargs="+",
        default=["title", "text"],
        help="CSV columns to concatenate for article text.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where ML artifacts will be saved.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    records = load_training_records(args.fake_data, args.true_data, args.text_columns)
    model, vectorizer, metrics = train_and_evaluate(
        records,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    label_counts = {
        LABEL_NAMES[label]: sum(record.label == label for record in records)
        for label in LABEL_NAMES
    }
    metadata = {
        "label_names": {str(label): name for label, name in LABEL_NAMES.items()},
        "fake_label": FAKE_LABEL,
        "true_label": TRUE_LABEL,
        "text_columns": args.text_columns,
        "preprocessing": {
            "version": PREPROCESSING_VERSION,
            "removed_source_markers": SOURCE_MARKERS,
        },
        "dataset": {
            "fake_path": str(args.fake_data),
            "true_path": str(args.true_data),
            "label_counts": label_counts,
        },
        "model": {
            "classes": [int(label) for label in model.classes_],
            "vectorizer_vocabulary_size": len(vectorizer.vocabulary_),
        },
        "evaluation": metrics,
    }
    save_artifacts(model, vectorizer, args.output_dir, metadata)


if __name__ == "__main__":
    main()
