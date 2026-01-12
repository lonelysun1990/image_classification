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
