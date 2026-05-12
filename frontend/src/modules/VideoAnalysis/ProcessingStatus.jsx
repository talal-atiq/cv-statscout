const STAGE_LABELS = {
  queued: "Queued",
  downloading_models: "Downloading models (first time only)...",
  extracting_frames: "Extracting frames...",
  detecting: "Detecting players & ball...",
  classifying_teams: "Classifying teams...",
  computing_metrics: "Computing analytics...",
  rendering: "Rendering output video...",
  done: "Complete!",
};

export default function ProcessingStatus({ uploadProgress, jobStatus }) {
  const status = jobStatus?.status || "queued";
  const progress = jobStatus?.progress || 0;
  const message = jobStatus?.status_message || "Initialising...";

  return (
    <div className="processing-status">
      <h2>Processing Match Video</h2>

      {uploadProgress < 100 && (
        <div>
          <p className="stage-label">Uploading video...</p>
          <p className="message">{uploadProgress}% uploaded</p>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${uploadProgress}%` }} />
          </div>
          <p className="progress-text">{uploadProgress}%</p>
        </div>
      )}

      {uploadProgress >= 100 && (
        <div>
          <p className="stage-label">{STAGE_LABELS[status] || status}</p>
          <p className="message">{message}</p>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
          <p className="progress-text">{progress}%</p>
        </div>
      )}
    </div>
  );
}
