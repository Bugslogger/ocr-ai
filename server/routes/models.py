"""Read-only model-cache inspection endpoints."""
from pathlib import Path
import os

from fastapi import APIRouter

from server.schemas import ModelStatus, ModelsStatusResponse

router = APIRouter(prefix="/models", tags=["Models"])

AI_ROOT = Path(__file__).resolve().parents[2] / "AI"
MODEL_SPECS = (
    ("baidu/Unlimited-OCR", "models--baidu--Unlimited-OCR", "Unlimited-OCR"),
    ("Qwen/Qwen2.5-7B-Instruct", "models--Qwen--Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct"),
)


def _huggingface_hub_dir() -> Path:
    configured_cache = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if configured_cache:
        return Path(configured_cache).expanduser()
    hf_home = Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "hub"


def _candidate_locations(cache_name: str, repository_name: str) -> list[Path]:
    project_caches = (AI_ROOT / "models", AI_ROOT / "model_cache", AI_ROOT / "huggingface" / "hub")
    candidates = [cache / cache_name for cache in project_caches]
    candidates.append(AI_ROOT / "models" / repository_name)
    candidates.append(_huggingface_hub_dir() / cache_name)
    return candidates


def _snapshot_dirs(location: Path) -> list[Path]:
    snapshots = location / "snapshots"
    if snapshots.is_dir():
        return [path for path in snapshots.iterdir() if path.is_dir()]
    return [location] if location.is_dir() else []


def _is_downloaded(snapshot: Path) -> bool:
    has_config = (snapshot / "config.json").is_file()
    has_weights = any(snapshot.glob("*.safetensors")) or any(snapshot.glob("pytorch_model*.bin"))
    return has_config and has_weights


@router.get("/status", response_model=ModelsStatusResponse)
async def model_status() -> ModelsStatusResponse:
    statuses: list[ModelStatus] = []
    for model_id, cache_name, repository_name in MODEL_SPECS:
        locations = _candidate_locations(cache_name, repository_name)
        downloaded_path = next(
            (snapshot for location in locations for snapshot in _snapshot_dirs(location) if _is_downloaded(snapshot)),
            None,
        )
        statuses.append(
            ModelStatus(
                model_id=model_id,
                downloaded=downloaded_path is not None,
                location=str(downloaded_path) if downloaded_path else None,
                locations_checked=[str(location) for location in locations],
            )
        )
    return ModelsStatusResponse(all_downloaded=all(model.downloaded for model in statuses), models=statuses)
