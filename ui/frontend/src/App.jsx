import { useEffect, useMemo, useRef, useState } from "react";
import {
  fetchModelTypes,
  fetchTestSamples,
  fetchTrainingRuns,
  fetchTrainedModels,
  getTrainingProgress,
  predictSamples,
  startTraining,
  staticSampleUrl,
} from "./api.js";

const TABS = {
  test: "Test Run",
  training: "Training",
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
  const [predictionError, setPredictionError] = useState("");
  const [trainingRuns, setTrainingRuns] = useState([]);
  const [trainedModels, setTrainedModels] = useState([]);
  const [status, setStatus] = useState({ type: "", message: "" });

  // Training state
  const [trainConfig, setTrainConfig] = useState({
    model_type: "",
    num_epochs: 5,
    learning_rate: 0.001,
    batch_size: 64,
  });
  const [trainStatus, setTrainStatus] = useState("idle"); // idle | running | completed | error
  const [trainProgress, setTrainProgress] = useState([]);
  const [trainError, setTrainError] = useState("");
  const sseRef = useRef(null);

  // Image preview for uploads
  const [uploadPreviews, setUploadPreviews] = useState([]);

  useEffect(() => {
    async function bootstrap() {
      try {
        const [models, samplesResponse, runs, trained] = await Promise.all([
          fetchModelTypes(),
          fetchTestSamples(),
          fetchTrainingRuns(),
          fetchTrainedModels(),
        ]);
        setModelTypes(models);
        setModelType(models[0] || "");
        setTrainConfig((prev) => ({ ...prev, model_type: models[0] || "" }));
        setTestSamples(samplesResponse.samples);
        setLabels(samplesResponse.labels);
        setTrainingRuns(runs);
        setTrainedModels(trained);
      } catch (error) {
        setStatus({ type: "error", message: error.message });
      }
    }
    bootstrap();
  }, []);

  // Generate previews when uploads change
  useEffect(() => {
    const previews = [];
    for (const file of uploads) {
      previews.push({ name: file.name, url: URL.createObjectURL(file) });
    }
    setUploadPreviews(previews);
    return () => previews.forEach((p) => URL.revokeObjectURL(p.url));
  }, [uploads]);

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
    if (!canPredict) return;
    setPredictionError("");
    const formData = new FormData();
    formData.append("model_type", modelType);
    formData.append("sample_ids", Array.from(selectedSamples).join(","));
    uploads.forEach((file) => formData.append("files", file));

    try {
      const response = await predictSamples(formData);
      if (response.error) {
        setPredictionError(response.error);
        setPredictions([]);
      } else {
        setPredictions(response.predictions);
        setStatus({ type: "success", message: "Prediction completed." });
      }
    } catch (error) {
      setStatus({ type: "error", message: error.message });
    }
  }

  async function handleStartTraining() {
    setTrainStatus("running");
    setTrainProgress([]);
    setTrainError("");

    try {
      const result = await startTraining(trainConfig);
      if (result.status === "error") {
        setTrainStatus("error");
        setTrainError(result.message);
        return;
      }

      // Connect to SSE for progress
      if (sseRef.current) sseRef.current.close();

      sseRef.current = getTrainingProgress(
        // onEpoch
        (epochData) => {
          setTrainProgress((prev) => [...prev, epochData]);
        },
        // onDone
        async () => {
          setTrainStatus("completed");
          // Refresh runs and models
          const [runs, trained] = await Promise.all([
            fetchTrainingRuns(),
            fetchTrainedModels(),
          ]);
          setTrainingRuns(runs);
          setTrainedModels(trained);
        },
        // onError
        (errData) => {
          setTrainStatus("error");
          setTrainError(errData.error || "Training failed.");
        }
      );
    } catch (error) {
      setTrainStatus("error");
      setTrainError(error.message);
    }
  }

  // Cleanup SSE on unmount
  useEffect(() => {
    return () => {
      if (sseRef.current) sseRef.current.close();
    };
  }, []);

  const latestEpoch = trainProgress.length > 0 ? trainProgress[trainProgress.length - 1] : null;

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <p className="eyebrow">Image Classification UI</p>
          <h1>Model Training &amp; Test Console</h1>
          <p className="subtitle">
            Train models on FashionMNIST, evaluate predictions, and view results
            with confidence scores.
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

      {/* ===== TEST TAB ===== */}
      {activeTab === "test" ? (
        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Test Run</h2>
              <p>
                Select a model, choose test samples, or upload your own images
                to verify predictions.
              </p>
            </div>
            <div className="select-group">
              <label htmlFor="model-select">Model</label>
              <select
                id="model-select"
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
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
                onChange={(e) =>
                  setUploads(Array.from(e.target.files || []))
                }
              />
              {uploadPreviews.length > 0 && (
                <div className="upload-previews">
                  {uploadPreviews.map((p) => (
                    <div key={p.name} className="upload-preview">
                      <img src={p.url} alt={p.name} />
                      <span>{p.name}</span>
                    </div>
                  ))}
                </div>
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

          {/* Prediction error */}
          {predictionError && (
            <div className="status error">{predictionError}</div>
          )}

          {/* Prediction Results */}
          <div className="card">
            <h3>Prediction Results</h3>
            {predictions.length > 0 ? (
              <div className="prediction-results">
                {predictions.map((pred, index) => (
                  <div key={`${pred.filename}-${index}`} className="prediction-card">
                    <div className="prediction-image">
                      {pred.image_base64 ? (
                        <img
                          src={`data:image/png;base64,${pred.image_base64}`}
                          alt={pred.filename}
                        />
                      ) : (
                        <div className="no-preview">No preview</div>
                      )}
                    </div>
                    <div className="prediction-details">
                      <div className="prediction-header">
                        <strong>{pred.predicted_label}</strong>
                        <span className="confidence-badge">
                          {pred.confidence}%
                        </span>
                      </div>
                      <p className="prediction-filename">{pred.filename}</p>
                      {pred.true_label && (
                        <p className="prediction-truth">
                          True: {pred.true_label}{" "}
                          {pred.correct !== null &&
                            (pred.correct ? (
                              <span className="correct">Correct</span>
                            ) : (
                              <span className="incorrect">Incorrect</span>
                            ))}
                        </p>
                      )}
                      <div className="top3">
                        {pred.top3 &&
                          pred.top3.map((item) => (
                            <div key={item.label} className="confidence-row">
                              <span className="conf-label">{item.label}</span>
                              <div className="confidence-bar-track">
                                <div
                                  className="confidence-bar-fill"
                                  style={{ width: `${item.confidence}%` }}
                                />
                              </div>
                              <span className="conf-value">
                                {item.confidence}%
                              </span>
                            </div>
                          ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="hint">No predictions yet. Choose samples to begin.</p>
            )}
          </div>
        </section>
      ) : (
        /* ===== TRAINING TAB ===== */
        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Training</h2>
              <p>
                Configure and train models on FashionMNIST with live progress
                tracking.
              </p>
            </div>
          </div>

          <div className="grid">
            {/* Training Config Form */}
            <div className="card">
              <h3>Train a Model</h3>
              <label>
                Model Type
                <select
                  value={trainConfig.model_type}
                  onChange={(e) =>
                    setTrainConfig((prev) => ({
                      ...prev,
                      model_type: e.target.value,
                    }))
                  }
                  disabled={trainStatus === "running"}
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
                  Epochs
                  <input
                    type="number"
                    min="1"
                    max="100"
                    value={trainConfig.num_epochs}
                    onChange={(e) =>
                      setTrainConfig((prev) => ({
                        ...prev,
                        num_epochs: parseInt(e.target.value) || 1,
                      }))
                    }
                    disabled={trainStatus === "running"}
                  />
                </label>
                <label>
                  Learning Rate
                  <input
                    type="number"
                    step="0.0001"
                    min="0.00001"
                    value={trainConfig.learning_rate}
                    onChange={(e) =>
                      setTrainConfig((prev) => ({
                        ...prev,
                        learning_rate: parseFloat(e.target.value) || 0.001,
                      }))
                    }
                    disabled={trainStatus === "running"}
                  />
                </label>
                <label>
                  Batch Size
                  <input
                    type="number"
                    min="1"
                    max="512"
                    value={trainConfig.batch_size}
                    onChange={(e) =>
                      setTrainConfig((prev) => ({
                        ...prev,
                        batch_size: parseInt(e.target.value) || 64,
                      }))
                    }
                    disabled={trainStatus === "running"}
                  />
                </label>
              </div>
              <button
                className="primary"
                type="button"
                onClick={handleStartTraining}
                disabled={trainStatus === "running"}
              >
                {trainStatus === "running" ? "Training..." : "Train Model"}
              </button>

              {/* Progress section */}
              {trainStatus === "running" && (
                <div className="train-progress">
                  <div className="progress-header">
                    <span>
                      Epoch {latestEpoch ? latestEpoch.epoch : 0} /{" "}
                      {trainConfig.num_epochs}
                    </span>
                    <span>
                      {latestEpoch
                        ? `${Math.round(
                            (latestEpoch.epoch / trainConfig.num_epochs) * 100
                          )}%`
                        : "0%"}
                    </span>
                  </div>
                  <div className="progress-bar-track">
                    <div
                      className="progress-bar-fill"
                      style={{
                        width: latestEpoch
                          ? `${(latestEpoch.epoch / trainConfig.num_epochs) * 100}%`
                          : "0%",
                      }}
                    />
                  </div>
                  {latestEpoch && (
                    <div className="epoch-metrics">
                      <span>Loss: {latestEpoch.train_loss.toFixed(4)}</span>
                      <span>Acc: {latestEpoch.train_acc.toFixed(2)}%</span>
                      <span>Val Loss: {latestEpoch.val_loss.toFixed(4)}</span>
                      <span>Val Acc: {latestEpoch.val_acc.toFixed(2)}%</span>
                    </div>
                  )}
                </div>
              )}

              {/* Completed summary */}
              {trainStatus === "completed" && latestEpoch && (
                <div className="train-complete">
                  <h4>Training Complete</h4>
                  <div className="epoch-metrics">
                    <span>Final Loss: {latestEpoch.train_loss.toFixed(4)}</span>
                    <span>Final Acc: {latestEpoch.train_acc.toFixed(2)}%</span>
                    <span>Val Loss: {latestEpoch.val_loss.toFixed(4)}</span>
                    <span>Val Acc: {latestEpoch.val_acc.toFixed(2)}%</span>
                  </div>
                </div>
              )}

              {/* Error */}
              {trainStatus === "error" && (
                <div className="status error">{trainError}</div>
              )}

              {/* Epoch history table */}
              {trainProgress.length > 0 && (
                <div className="epoch-history">
                  <h4>Epoch History</h4>
                  <table>
                    <thead>
                      <tr>
                        <th>Epoch</th>
                        <th>Train Loss</th>
                        <th>Train Acc</th>
                        <th>Val Loss</th>
                        <th>Val Acc</th>
                      </tr>
                    </thead>
                    <tbody>
                      {trainProgress.map((ep) => (
                        <tr key={ep.epoch}>
                          <td>{ep.epoch}</td>
                          <td>{ep.train_loss.toFixed(4)}</td>
                          <td>{ep.train_acc.toFixed(2)}%</td>
                          <td>{ep.val_loss.toFixed(4)}</td>
                          <td>{ep.val_acc.toFixed(2)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Saved Runs */}
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

              {/* Trained models */}
              {trainedModels.length > 0 && (
                <>
                  <h3>Available Checkpoints</h3>
                  <div className="runs">
                    {trainedModels.map((m) => (
                      <article key={m.filename} className="run">
                        <header>
                          <strong>{m.model_type.toUpperCase()}</strong>
                          <span>{m.filename}</span>
                        </header>
                        {m.best_val_acc != null && (
                          <p>Best Val Acc: {m.best_val_acc.toFixed(2)}%</p>
                        )}
                      </article>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
