"""Model repository downloads and read-only cache inspection."""
from pathlib import Path
import json
import logging

from fastapi import APIRouter, HTTPException

from server.schemas import ModelStatus, ModelsStatusResponse, ModelDownloadResponse
from AI.model import MODEL_DEFAULTS, model_spec, model_cache_dir

router = APIRouter(prefix="/models", tags=["Models"])

AI_ROOT = Path(__file__).resolve().parents[2] / "AI"
logger = logging.getLogger(__name__)


def _huggingface_hub_dir() -> Path:
    return model_cache_dir()


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
    if not (snapshot / "config.json").is_file():
        return False
    for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = snapshot / filename
        if index.is_file():
            try:
                shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
                return bool(shards) and all((snapshot / shard).is_file() for shard in shards)
            except (ValueError, KeyError, TypeError, AttributeError, OSError):
                return False
    return (snapshot / "model.safetensors").is_file() or (snapshot / "pytorch_model.bin").is_file()


@router.post("/download", response_model=ModelDownloadResponse,
             summary="Download both model repositories to disk without loading models",
             responses={502: {"description": "Model repository download failed"}})
def download_models() -> ModelDownloadResponse:
    from AI.model import download_model_files

    try:
        return ModelDownloadResponse(models=download_model_files())
    except Exception as exc:
        logger.exception("Model repository download failed")
        raise HTTPException(status_code=502, detail="Model download failed. Check the server logs for details.") from exc


@router.get("/status", response_model=ModelsStatusResponse)
async def model_status() -> ModelsStatusResponse:
    statuses: list[ModelStatus] = []
    for kind in MODEL_DEFAULTS:
        model_id, _ = model_spec(kind)
        cache_name = "models--" + model_id.replace("/", "--")
        repository_name = model_id.rsplit("/", 1)[-1]
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
