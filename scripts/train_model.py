import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FAKE_PATH = ROOT_DIR / "data" / "Fake.csv"
DEFAULT_TRUE_PATH = ROOT_DIR / "data" / "True.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "models_ml"
MODEL_FILENAME = "fake_news_model.pkl"
VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"
FAKE_LABEL = 0
TRUE_LABEL = 1


@dataclass(frozen=True)
class ArticleRecord:
    text: str
    label: int


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_labeled_articles(
    csv_path: Path,
    label: int,
    text_column: str,
) -> list[ArticleRecord]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    records: list[ArticleRecord] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {csv_path}")
        if text_column not in reader.fieldnames:
            available_columns = ", ".join(reader.fieldnames)
            raise ValueError(
                f"Missing text column {text_column!r} in {csv_path}. "
                f"Available columns: {available_columns}"
            )

        for row in reader:
            text = (row.get(text_column) or "").strip()
            if text:
                records.append(ArticleRecord(text=text, label=label))

    if not records:
        raise ValueError(f"No usable article text found in {csv_path}")

    return records


def load_training_records(
    fake_path: Path,
    true_path: Path,
    text_column: str,
) -> list[ArticleRecord]:
    fake_records = load_labeled_articles(fake_path, FAKE_LABEL, text_column)
    true_records = load_labeled_articles(true_path, TRUE_LABEL, text_column)
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
        max_df=0.95,
    )


def build_model() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )


def print_metrics(y_true: list[int], y_pred: list[int]) -> None:
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")


def train_and_evaluate(
    records: list[ArticleRecord],
    test_size: float,
    random_state: int,
) -> tuple[LogisticRegression, TfidfVectorizer]:
    x_train, x_test, y_train, y_test = split_records(records, test_size, random_state)
    x_train = [preprocess_text(text) for text in x_train]
    x_test = [preprocess_text(text) for text in x_test]

    vectorizer = build_vectorizer()
    x_train_features = vectorizer.fit_transform(x_train)
    x_test_features = vectorizer.transform(x_test)

    model = build_model()
    model.fit(x_train_features, y_train)
    y_pred = model.predict(x_test_features)

    print_metrics(y_test, y_pred)

    all_texts = [preprocess_text(record.text) for record in records]
    all_labels = [record.label for record in records]
    final_vectorizer = build_vectorizer()
    all_features = final_vectorizer.fit_transform(all_texts)
    final_model = build_model()
    final_model.fit(all_features, all_labels)

    return final_model, final_vectorizer


def save_artifacts(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / MODEL_FILENAME
    vectorizer_path = output_dir / VECTORIZER_FILENAME

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Saved model to: {model_path}")
    print(f"Saved vectorizer to: {vectorizer_path}")


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
        "--text-column",
        default="text",
        help="CSV column containing article text.",
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
    args = parse_args()
    records = load_training_records(args.fake_data, args.true_data, args.text_column)
    model, vectorizer = train_and_evaluate(
        records,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    save_artifacts(model, vectorizer, args.output_dir)


if __name__ == "__main__":
    main()
