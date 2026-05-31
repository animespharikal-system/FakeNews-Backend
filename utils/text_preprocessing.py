import re


PREPROCESSING_VERSION = "2026-05-31-source-normalized-v1"
SOURCE_MARKERS = ("reuters", "associated press")


def preprocess_text(text: str) -> str:
    """Normalize article text consistently for training and inference."""
    text = re.sub(
        r"(?i)^\s*[A-Z][A-Z\s.,/-]{1,80}\((?:REUTERS|AP)\)\s*[-:]\s*",
        " ",
        text,
    )
    text = re.sub(r"(?i)\b(?:reuters|associated press)\b", " ", text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
