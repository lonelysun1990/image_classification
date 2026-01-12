import { useEffect, useMemo, useState } from "react";
import {
  createTrainingRun,
  fetchModelTypes,
  fetchTestSamples,
  fetchTrainingRuns,
  predictSamples,
  staticSampleUrl
} from "./api.js";

const TABS = {
  test: "Test Run",
  training: "Training"
};

const emptyMetrics = {
  train_accuracy: "",
  train_loss: "",
  val_accuracy: "",
  val_loss: ""
};

export default function App() {
  const [activeTab, setActiveTab] = useState("test");
  const [modelTypes, setModelTypes] = useState([]);
  const [modelType, setModelType] = useState("");
  const [testSamples, setTestSamples] = useState([]);
  const [labels, setLabels] = useState([]);
  const [selectedSamples, setSelectedSamples] = useState(new Set());
  const [uploads, setUploads] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [trainingRuns, setTrainingRuns] = useState([]);
  const [trainingForm, setTrainingForm] = useState({
    model_type: "",
    notes: "",
    ...emptyMetrics
  });
  const [status, setStatus] = useState({ type: "", message: "" });

  useEffect(() => {
    async function bootstrap() {
      try {
        const [models, samplesResponse, runs] = await Promise.all([
          fetchModelTypes(),
          fetchTestSamples(),
          fetchTrainingRuns()
        ]);
        setModelTypes(models);
        setModelType(models[0] || "");
        setTrainingForm((prev) => ({
          ...prev,
          model_type: models[0] || ""
        }));
        setTestSamples(samplesResponse.samples);
        setLabels(samplesResponse.labels);
        setTrainingRuns(runs);
      } catch (error) {
        setStatus({ type: "error", message: error.message });
      }
    }

    bootstrap();
  }, []);

  const canPredict = useMemo(
    () => modelType && (selectedSamples.size > 0 || uploads.length > 0),
    [modelType, selectedSamples, uploads]
  );

  function toggleSample(sampleId) {
    setSelectedSamples((prev) => {
      const next = new Set(prev);
      if (next.has(sampleId)) {
        next.delete(sampleId);
      } else {
        next.add(sampleId);
      }
      return next;
    });
  }

  async function handlePredict() {
    if (!canPredict) {
      return;
    }
    const formData = new FormData();
    formData.append("model_type", modelType);
    formData.append("sample_ids", Array.from(selectedSamples).join(","));
    uploads.forEach((file) => {
      formData.append("files", file);
    });

    try {
      const response = await predictSamples(formData);
      setPredictions(response.predictions);
      setStatus({ type: "success", message: "Prediction completed." });
    } catch (error) {
      setStatus({ type: "error", message: error.message });
    }
  }

  async function handleTrainingSubmit(event) {
    event.preventDefault();
    const payload = {
      model_type: trainingForm.model_type,
      train_accuracy: Number(trainingForm.train_accuracy),
      train_loss: Number(trainingForm.train_loss),
      val_accuracy: Number(trainingForm.val_accuracy),
      val_loss: Number(trainingForm.val_loss),
      notes: trainingForm.notes || null
    };

    try {
      const run = await createTrainingRun(payload);
      setTrainingRuns((prev) => [run, ...prev]);
      setTrainingForm((prev) => ({
        ...prev,
        notes: "",
        ...emptyMetrics
      }));
      setStatus({ type: "success", message: "Training run logged." });
    } catch (error) {
      setStatus({ type: "error", message: error.message });
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <p className="eyebrow">Image Classification UI</p>
          <h1>Model Training &amp; Test Console</h1>
          <p className="subtitle">
            Track model versions, evaluate predictions, and benchmark against test
            labels.
          </p>
        </div>
        <nav className="tabs">
          {Object.entries(TABS).map(([key, label]) => (
            <button
              key={key}
              className={key === activeTab ? "tab active" : "tab"}
              type="button"
              onClick={() => setActiveTab(key)}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>

      {status.message ? (
        <div className={`status ${status.type}`}>{status.message}</div>
      ) : null}

      {activeTab === "test" ? (
        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Test Run</h2>
              <p>
                Select a model, choose test samples, or upload your own images to
                verify predictions.
              </p>
            </div>
            <div className="select-group">
              <label htmlFor="model-select">Model</label>
              <select
                id="model-select"
                value={modelType}
                onChange={(event) => setModelType(event.target.value)}
              >
                {modelTypes.map((model) => (
                  <option key={model} value={model}>
                    {model.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="grid">
            <div className="card">
              <h3>Test Dataset Samples</h3>
              <div className="samples">
                {testSamples.map((sample) => (
                  <label key={sample.id} className="sample">
                    <input
                      type="checkbox"
                      checked={selectedSamples.has(sample.id)}
                      onChange={() => toggleSample(sample.id)}
                    />
                    <img
                      src={staticSampleUrl(sample.filename)}
                      alt={`Sample ${sample.id}`}
                    />
                    <div>
                      <strong>{sample.label}</strong>
                      <span>{sample.filename}</span>
                    </div>
                  </label>
                ))}
              </div>
              <p className="hint">Labels in dataset: {labels.join(", ")}</p>
            </div>

            <div className="card">
              <h3>Upload Images</h3>
              <p>Select one or more files to evaluate.</p>
              <input
                type="file"
                multiple
                accept="image/*"
                onChange={(event) =>
                  setUploads(Array.from(event.target.files || []))
                }
              />
              {uploads.length ? (
                <ul className="file-list">
                  {uploads.map((file) => (
                    <li key={file.name}>{file.name}</li>
                  ))}
                </ul>
              ) : (
                <p className="hint">No uploads selected.</p>
              )}
              <button
                className="primary"
                type="button"
                onClick={handlePredict}
                disabled={!canPredict}
              >
                Run Prediction
              </button>
            </div>
          </div>

          <div className="card">
            <h3>Prediction Results</h3>
            {predictions.length ? (
              <table>
                <thead>
                  <tr>
                    <th>Image</th>
                    <th>Predicted Label</th>
                    <th>True Label</th>
                    <th>Correct?</th>
                  </tr>
                </thead>
                <tbody>
                  {predictions.map((prediction, index) => (
                    <tr key={`${prediction.filename}-${index}`}>
                      <td>{prediction.filename}</td>
                      <td>{prediction.predicted_label}</td>
                      <td>{prediction.true_label || "Uploaded"}</td>
                      <td>
                        {prediction.correct === null
                          ? "—"
                          : prediction.correct
                          ? "Yes"
                          : "No"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="hint">No predictions yet. Choose samples to begin.</p>
            )}
          </div>
        </section>
      ) : (
        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Training Runs</h2>
              <p>
                Track each model version with metrics and timestamps to keep a
                complete audit trail.
              </p>
            </div>
          </div>

          <div className="grid">
            <form className="card" onSubmit={handleTrainingSubmit}>
              <h3>Log a Training Run</h3>
              <label>
                Model Type
                <select
                  value={trainingForm.model_type}
                  onChange={(event) =>
                    setTrainingForm((prev) => ({
                      ...prev,
                      model_type: event.target.value
                    }))
                  }
                >
                  {modelTypes.map((model) => (
                    <option key={model} value={model}>
                      {model.toUpperCase()}
                    </option>
                  ))}
                </select>
              </label>
              <div className="metrics">
                <label>
                  Train Accuracy
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    max="1"
                    value={trainingForm.train_accuracy}
                    onChange={(event) =>
                      setTrainingForm((prev) => ({
                        ...prev,
                        train_accuracy: event.target.value
                      }))
                    }
                    required
                  />
                </label>
                <label>
                  Train Loss
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    value={trainingForm.train_loss}
                    onChange={(event) =>
                      setTrainingForm((prev) => ({
                        ...prev,
                        train_loss: event.target.value
                      }))
                    }
                    required
                  />
                </label>
                <label>
                  Validation Accuracy
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    max="1"
                    value={trainingForm.val_accuracy}
                    onChange={(event) =>
                      setTrainingForm((prev) => ({
                        ...prev,
                        val_accuracy: event.target.value
                      }))
                    }
                    required
                  />
                </label>
                <label>
                  Validation Loss
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    value={trainingForm.val_loss}
                    onChange={(event) =>
                      setTrainingForm((prev) => ({
                        ...prev,
                        val_loss: event.target.value
                      }))
                    }
                    required
                  />
                </label>
              </div>
              <label>
                Notes
                <textarea
                  rows="3"
                  value={trainingForm.notes}
                  onChange={(event) =>
                    setTrainingForm((prev) => ({
                      ...prev,
                      notes: event.target.value
                    }))
                  }
                />
              </label>
              <button className="primary" type="submit">
                Save Training Run
              </button>
            </form>

            <div className="card">
              <h3>Saved Runs</h3>
              <div className="runs">
                {trainingRuns.map((run) => (
                  <article key={run.run_id} className="run">
                    <header>
                      <strong>{run.version}</strong>
                      <span>{run.model_type.toUpperCase()}</span>
                    </header>
                    <p className="muted">{run.trained_at}</p>
                    <ul>
                      <li>Train Acc: {run.train_accuracy}</li>
                      <li>Train Loss: {run.train_loss}</li>
                      <li>Val Acc: {run.val_accuracy}</li>
                      <li>Val Loss: {run.val_loss}</li>
                    </ul>
                    {run.notes ? <p>{run.notes}</p> : null}
                    <p className="muted">ID: {run.run_id}</p>
                  </article>
                ))}
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
