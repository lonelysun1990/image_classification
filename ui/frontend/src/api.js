const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function fetchModelTypes() {
  const response = await fetch(`${API_BASE}/api/model-types`);
  if (!response.ok) {
    throw new Error("Failed to load model types.");
  }
  return response.json();
}

export async function fetchTrainingRuns() {
  const response = await fetch(`${API_BASE}/api/training-runs`);
  if (!response.ok) {
    throw new Error("Failed to load training runs.");
  }
  return response.json();
}

export async function createTrainingRun(payload) {
  const response = await fetch(`${API_BASE}/api/training-runs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    throw new Error("Failed to create training run.");
  }
  return response.json();
}

export async function fetchTestSamples() {
  const response = await fetch(`${API_BASE}/api/test-samples`);
  if (!response.ok) {
    throw new Error("Failed to load test samples.");
  }
  return response.json();
}

export async function predictSamples(formData) {
  const response = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    body: formData
  });
  if (!response.ok) {
    throw new Error("Prediction request failed.");
  }
  return response.json();
}

export function staticSampleUrl(filename) {
  return `${API_BASE}/static/${filename}`;
}

export async function startTraining(config) {
  const response = await fetch(`${API_BASE}/api/train`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(config)
  });
  if (!response.ok) {
    throw new Error("Failed to start training.");
  }
  return response.json();
}

export function getTrainingProgress(onEpoch, onDone, onError) {
  const eventSource = new EventSource(`${API_BASE}/api/train/progress`);

  eventSource.addEventListener("epoch", (event) => {
    const data = JSON.parse(event.data);
    onEpoch(data);
  });

  eventSource.addEventListener("done", (event) => {
    const data = JSON.parse(event.data);
    onDone(data);
    eventSource.close();
  });

  eventSource.addEventListener("error", (event) => {
    if (event.data) {
      const data = JSON.parse(event.data);
      onError(data);
    } else {
      onError({ error: "Connection lost" });
    }
    eventSource.close();
  });

  eventSource.addEventListener("idle", () => {
    eventSource.close();
  });

  return eventSource;
}

export async function getTrainStatus() {
  const response = await fetch(`${API_BASE}/api/train/status`);
  if (!response.ok) {
    throw new Error("Failed to get training status.");
  }
  return response.json();
}

export async function fetchTrainedModels() {
  const response = await fetch(`${API_BASE}/api/trained-models`);
  if (!response.ok) {
    throw new Error("Failed to load trained models.");
  }
  return response.json();
}
