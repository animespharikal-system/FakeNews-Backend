from fastapi import APIRouter


router = APIRouter(tags=["Status"])


@router.get("/")
def read_root() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "Fake News Detection API is running",
    }
