import { useState, useEffect, useRef } from "react";
import UploadZone from "./UploadZone.jsx";
import ProcessingStatus from "./ProcessingStatus.jsx";
import ResultsViewer from "./ResultsViewer.jsx";
import { uploadVideo, getJobStatus } from "./api.js";

const POLLING_INTERVAL_MS = 2000;

export default function VideoAnalysis() {
  const [phase, setPhase] = useState("upload"); // upload | processing | results
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const pollingRef = useRef(null);

  const handleUpload = async (file) => {
    setPhase("processing");
    setUploadProgress(0);
    try {
      const res = await uploadVideo(file, setUploadProgress);
      setJobId(res.job_id);
    } catch (err) {
      alert("Upload failed: " + err.message);
      setPhase("upload");
    }
  };

  useEffect(() => {
    if (!jobId) return;

    const poll = async () => {
      try {
        const status = await getJobStatus(jobId);
        setJobStatus(status);
        if (status.status === "done") {
          clearInterval(pollingRef.current);
          setPhase("results");
        } else if (status.status === "failed") {
          clearInterval(pollingRef.current);
          alert("Processing failed: " + status.error);
          setPhase("upload");
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    };

    pollingRef.current = setInterval(poll, POLLING_INTERVAL_MS);
    poll();

    return () => clearInterval(pollingRef.current);
  }, [jobId]);

  return (
    <div className="video-analysis-module">
      {phase === "upload" && <UploadZone onUpload={handleUpload} />}
      {phase === "processing" && (
        <ProcessingStatus uploadProgress={uploadProgress} jobStatus={jobStatus} />
      )}
      {phase === "results" && jobStatus && (
        <ResultsViewer
          videoUrl={jobStatus.output_video_url}
          mainVideoUrl={jobStatus.main_video_url}
          pitchVideoUrl={jobStatus.pitch_video_url}
          voronoiVideoUrl={jobStatus.voronoi_video_url}
          analytics={jobStatus.analytics}
        />
      )}
    </div>
  );
}
