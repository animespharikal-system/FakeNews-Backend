import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


load_dotenv()


def _parse_origins(value: str) -> list[str]:
    return [origin.strip() for origin in value.split(",") if origin.strip()]


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Fake News Detection API")
    allowed_origins: list[str] = field(init=False)

    def __post_init__(self) -> None:
        origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,"
    "http://127.0.0.1:3000,"
    "http://localhost:5173,"
    "http://127.0.0.1:5173,"
    "http://localhost:5174,"
    "http://127.0.0.1:5174"
)
        object.__setattr__(self, "allowed_origins", _parse_origins(origins))


settings = Settings()
