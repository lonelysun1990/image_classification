from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from torchvision import transforms

from data import FashionMNISTDataset
from data.datasets import FASHION_MNIST_CLASSES
from models import get_model, list_models
from training.trainer import Trainer

from .storage import read_json, write_json

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = BASE_DIR / "sample_data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
TRAINING_RUNS_PATH = DATA_DIR / "training_runs.json"
TEST_SAMPLES_PATH = DATA_DIR / "test_samples.json"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TrainingRunIn(BaseModel):
    model_type: str = Field(..., min_length=1)
    train_accuracy: float = Field(..., ge=0, le=1)
    train_loss: float = Field(..., ge=0)
    val_accuracy: float = Field(..., ge=0, le=1)
    val_loss: float = Field(..., ge=0)
    notes: Optional[str] = None


class TrainingRunOut(TrainingRunIn):
    run_id: str
    version: str
    trained_at: str


class TrainRequest(BaseModel):
    model_type: str = Field(..., min_length=1)
    num_epochs: int = Field(default=5, ge=1, le=100)
    learning_rate: float = Field(default=1e-3, gt=0)
    batch_size: int = Field(default=64, ge=1, le=512)


class PredictionResult(BaseModel):
    sample_id: Optional[str]
    filename: str
    true_label: Optional[str]
    predicted_label: str
    confidence: float
    top3: List[Dict[str, Any]]
    correct: Optional[bool]
    image_base64: Optional[str] = None


# ---------------------------------------------------------------------------
# Training state (shared across threads)
# ---------------------------------------------------------------------------

_train_lock = threading.Lock()
_train_state: Dict[str, Any] = {
    "status": "idle",       # idle | running | completed | error
    "model_type": None,
    "progress": [],         # list of epoch dicts
    "current_epoch": 0,
    "total_epochs": 0,
    "error": None,
    "checkpoint_path": None,
}


def _reset_train_state() -> None:
    with _train_lock:
        _train_state.update({
            "status": "idle",
            "model_type": None,
            "progress": [],
            "current_epoch": 0,
            "total_epochs": 0,
            "error": None,
            "checkpoint_path": None,
        })


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Image Classification UI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=SAMPLE_DIR), name="static")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_training_runs() -> List[Dict[str, str]]:
    return read_json(TRAINING_RUNS_PATH, default=[])


def load_test_samples() -> List[Dict[str, str]]:
    return read_json(TEST_SAMPLES_PATH, default=[])


def generate_run_id(model_type: str, existing_runs: List[Dict[str, str]]) -> str:
    slug = model_type.lower().replace(" ", "-")
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    count = sum(1 for run in existing_runs if run["model_type"].lower() == model_type.lower()) + 1
    return f"{slug}-run-{timestamp}-{count:02d}"


def generate_version(model_type: str, existing_runs: List[Dict[str, str]]) -> str:
    count = sum(1 for run in existing_runs if run["model_type"].lower() == model_type.lower()) + 1
    return f"{model_type.upper()}-v{count:03d}"


def available_labels(samples: List[Dict[str, str]]) -> List[str]:
    labels = sorted({sample["label"] for sample in samples})
    return labels or ["class_a", "class_b", "class_c"]


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _inference_transform():
    """Transform for inference: resize to 28x28, grayscale, normalize."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def _image_to_base64(img: Image.Image) -> str:
    """Convert a PIL image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _load_model_for_inference(checkpoint_path: str) -> torch.nn.Module:
    """Load a model from a checkpoint file."""
    device = _get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type", "cnn")
    model = get_model(model_type)
    # CNN models use lazy FC init; run a dummy forward pass to build all layers
    # before loading the state dict.
    dummy = torch.zeros(1, 1, 28, 28, device=device)
    model.to(device)
    with torch.no_grad():
        model(dummy)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Training background worker
# ---------------------------------------------------------------------------

def _run_training(config: TrainRequest) -> None:
    """Run training in a background thread."""
    try:
        with _train_lock:
            _train_state["status"] = "running"
            _train_state["model_type"] = config.model_type
            _train_state["total_epochs"] = config.num_epochs
            _train_state["progress"] = []
            _train_state["current_epoch"] = 0
            _train_state["error"] = None
            _train_state["checkpoint_path"] = None

        device = _get_device()
        model = get_model(config.model_type)

        dataset = FashionMNISTDataset(
            data_dir=str(PROJECT_ROOT / "data" / "raw"),
            augmentation_level="standard",
        )
        loaders = dataset.get_dataloaders(
            batch_size=config.batch_size,
            num_workers=0,
        )

        trainer = Trainer(model=model, device=device)

        def on_progress(info: Dict[str, Any]) -> None:
            with _train_lock:
                _train_state["current_epoch"] = info["epoch"]
                _train_state["progress"].append(info)

        checkpoint_name = f"{config.model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pt"
        checkpoint_path = str(CHECKPOINT_DIR / checkpoint_name)

        trainer.fit(
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            num_epochs=config.num_epochs,
            lr=config.learning_rate,
            use_wandb=False,
            save_best_model=True,
            model_save_path=checkpoint_path,
            verbose=True,
            progress_callback=on_progress,
        )

        # Save final checkpoint with model_type metadata
        final_checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_type": config.model_type,
            "epoch": config.num_epochs,
            "history": trainer.history,
            "best_val_acc": trainer.best_val_acc,
            "config": {
                "num_epochs": config.num_epochs,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
            },
        }
        torch.save(final_checkpoint, checkpoint_path)

        # Log the training run
        runs = load_training_runs()
        run_id = generate_run_id(config.model_type, runs)
        version = generate_version(config.model_type, runs)
        final_progress = _train_state["progress"][-1] if _train_state["progress"] else {}
        record = TrainingRunOut(
            run_id=run_id,
            version=version,
            trained_at=datetime.utcnow().isoformat() + "Z",
            model_type=config.model_type,
            train_accuracy=round(final_progress.get("train_acc", 0) / 100, 4),
            train_loss=round(final_progress.get("train_loss", 0), 4),
            val_accuracy=round(final_progress.get("val_acc", 0) / 100, 4),
            val_loss=round(final_progress.get("val_loss", 0), 4),
            notes=f"Trained {config.num_epochs} epochs, lr={config.learning_rate}, batch={config.batch_size}",
        )
        runs.insert(0, record.dict())
        write_json(TRAINING_RUNS_PATH, runs)

        with _train_lock:
            _train_state["status"] = "completed"
            _train_state["checkpoint_path"] = checkpoint_path

    except Exception as exc:
        with _train_lock:
            _train_state["status"] = "error"
            _train_state["error"] = str(exc)


# ---------------------------------------------------------------------------
# Existing endpoints
# ---------------------------------------------------------------------------

@app.get("/api/model-types")
async def get_model_types() -> List[str]:
    return list_models()


@app.get("/api/training-runs")
async def get_training_runs() -> List[TrainingRunOut]:
    runs = load_training_runs()
    return [TrainingRunOut(**run) for run in runs]


@app.post("/api/training-runs")
async def create_training_run(payload: TrainingRunIn) -> TrainingRunOut:
    runs = load_training_runs()
    run_id = generate_run_id(payload.model_type, runs)
    version = generate_version(payload.model_type, runs)
    trained_at = datetime.utcnow().isoformat() + "Z"
    record = TrainingRunOut(
        run_id=run_id,
        version=version,
        trained_at=trained_at,
        **payload.dict(),
    )
    runs.insert(0, record.dict())
    write_json(TRAINING_RUNS_PATH, runs)
    return record


@app.get("/api/test-samples")
async def get_test_samples() -> Dict[str, Any]:
    samples = load_test_samples()
    return {
        "samples": samples,
        "labels": available_labels(samples),
    }


# ---------------------------------------------------------------------------
# Training endpoints
# ---------------------------------------------------------------------------

@app.post("/api/train")
async def start_training(req: TrainRequest) -> Dict[str, str]:
    with _train_lock:
        if _train_state["status"] == "running":
            return {"status": "error", "message": "Training is already in progress."}

    _reset_train_state()
    thread = threading.Thread(target=_run_training, args=(req,), daemon=True)
    thread.start()
    return {"status": "started", "message": f"Training {req.model_type} for {req.num_epochs} epochs."}


@app.get("/api/train/status")
async def train_status() -> Dict[str, Any]:
    with _train_lock:
        return {
            "status": _train_state["status"],
            "model_type": _train_state["model_type"],
            "current_epoch": _train_state["current_epoch"],
            "total_epochs": _train_state["total_epochs"],
            "error": _train_state["error"],
            "checkpoint_path": _train_state["checkpoint_path"],
        }


@app.get("/api/train/progress")
async def train_progress():
    """SSE endpoint that streams epoch progress as JSON events."""
    async def event_generator():
        last_sent = 0
        while True:
            with _train_lock:
                status = _train_state["status"]
                progress = list(_train_state["progress"])
                total = _train_state["total_epochs"]
                error = _train_state["error"]

            # Send any new epoch data
            for entry in progress[last_sent:]:
                yield {"event": "epoch", "data": json.dumps(entry)}
            last_sent = len(progress)

            if status == "completed":
                yield {"event": "done", "data": json.dumps({"status": "completed", "total_epochs": total})}
                break
            if status == "error":
                yield {"event": "error", "data": json.dumps({"status": "error", "error": error})}
                break
            if status == "idle":
                yield {"event": "idle", "data": json.dumps({"status": "idle"})}
                break

            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@app.get("/api/trained-models")
async def get_trained_models() -> List[Dict[str, Any]]:
    """List available trained model checkpoints."""
    models = []
    if CHECKPOINT_DIR.exists():
        for f in sorted(CHECKPOINT_DIR.glob("*.pt"), key=os.path.getmtime, reverse=True):
            try:
                checkpoint = torch.load(f, map_location="cpu", weights_only=False)
                models.append({
                    "filename": f.name,
                    "path": str(f),
                    "model_type": checkpoint.get("model_type", "unknown"),
                    "best_val_acc": checkpoint.get("best_val_acc", None),
                    "config": checkpoint.get("config", {}),
                })
            except Exception:
                continue
    return models


# ---------------------------------------------------------------------------
# Prediction endpoint (real inference)
# ---------------------------------------------------------------------------

@app.post("/api/predict")
async def predict(
    model_type: str = Form(...),
    sample_ids: str = Form(""),
    files: List[UploadFile] = File(default=[]),
    checkpoint: str = Form(""),
) -> Dict[str, Any]:
    samples = load_test_samples()
    sample_lookup = {sample["id"]: sample for sample in samples}
    results: List[Dict[str, Any]] = []

    # Determine which checkpoint to use
    checkpoint_path = None
    if checkpoint:
        checkpoint_path = checkpoint
    else:
        # Find latest checkpoint matching model_type
        if CHECKPOINT_DIR.exists():
            matching = sorted(
                [f for f in CHECKPOINT_DIR.glob(f"{model_type}_*.pt")],
                key=os.path.getmtime,
                reverse=True,
            )
            if matching:
                checkpoint_path = str(matching[0])

    if not checkpoint_path or not Path(checkpoint_path).exists():
        return {
            "predictions": [],
            "error": f"No trained checkpoint found for model '{model_type}'. Please train a model first.",
        }

    device = _get_device()
    model = _load_model_for_inference(checkpoint_path)
    transform = _inference_transform()

    # Process sample images
    requested_ids = [item for item in sample_ids.split(",") if item]
    for sample_id in requested_ids:
        sample = sample_lookup.get(sample_id)
        if not sample:
            continue

        img_path = SAMPLE_DIR / sample["filename"]
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        img_b64 = _image_to_base64(img)
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]

        top3_indices = probs.argsort(descending=True)[:3]
        top3 = [
            {"label": FASHION_MNIST_CLASSES[i], "confidence": round(probs[i].item() * 100, 2)}
            for i in top3_indices
        ]
        pred_idx = top3_indices[0].item()
        predicted_label = FASHION_MNIST_CLASSES[pred_idx]

        results.append(PredictionResult(
            sample_id=sample_id,
            filename=sample["filename"],
            true_label=sample.get("label"),
            predicted_label=predicted_label,
            confidence=round(probs[pred_idx].item() * 100, 2),
            top3=top3,
            correct=predicted_label == sample.get("label") if sample.get("label") else None,
            image_base64=img_b64,
        ).dict())

    # Process uploaded files
    for file in files:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_b64 = _image_to_base64(img)
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]

        top3_indices = probs.argsort(descending=True)[:3]
        top3 = [
            {"label": FASHION_MNIST_CLASSES[i], "confidence": round(probs[i].item() * 100, 2)}
            for i in top3_indices
        ]
        pred_idx = top3_indices[0].item()
        predicted_label = FASHION_MNIST_CLASSES[pred_idx]

        results.append(PredictionResult(
            sample_id=None,
            filename=file.filename,
            true_label=None,
            predicted_label=predicted_label,
            confidence=round(probs[pred_idx].item() * 100, 2),
            top3=top3,
            correct=None,
            image_base64=img_b64,
        ).dict())

    return {"predictions": results}
