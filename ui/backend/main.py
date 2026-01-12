from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from models import list_models

from .storage import read_json, write_json

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = BASE_DIR / "sample_data"
TRAINING_RUNS_PATH = DATA_DIR / "training_runs.json"
TEST_SAMPLES_PATH = DATA_DIR / "test_samples.json"


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


class PredictionResult(BaseModel):
    sample_id: Optional[str]
    filename: str
    true_label: Optional[str]
    predicted_label: str
    correct: Optional[bool]


app = FastAPI(title="Image Classification UI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory=SAMPLE_DIR), name="static")


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


def stable_prediction(model_type: str, filename: str, labels: List[str]) -> str:
    digest = sha256(f"{model_type}:{filename}".encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(labels)
    return labels[index]


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
async def get_test_samples() -> Dict[str, List[Dict[str, str]]]:
    samples = load_test_samples()
    return {
        "samples": samples,
        "labels": available_labels(samples),
    }


@app.post("/api/predict")
async def predict(
    model_type: str = Form(...),
    sample_ids: str = Form(""),
    files: List[UploadFile] = File(default=[]),
) -> Dict[str, List[PredictionResult]]:
    samples = load_test_samples()
    labels = available_labels(samples)
    sample_lookup = {sample["id"]: sample for sample in samples}
    results: List[PredictionResult] = []

    requested_ids = [item for item in sample_ids.split(",") if item]
    for sample_id in requested_ids:
        sample = sample_lookup.get(sample_id)
        if not sample:
            continue
        predicted = stable_prediction(model_type, sample["filename"], labels)
        results.append(
            PredictionResult(
                sample_id=sample_id,
                filename=sample["filename"],
                true_label=sample["label"],
                predicted_label=predicted,
                correct=predicted == sample["label"],
            )
        )

    for file in files:
        predicted = stable_prediction(model_type, file.filename, labels)
        results.append(
            PredictionResult(
                sample_id=None,
                filename=file.filename,
                true_label=None,
                predicted_label=predicted,
                correct=None,
            )
        )

    return {"predictions": results}
