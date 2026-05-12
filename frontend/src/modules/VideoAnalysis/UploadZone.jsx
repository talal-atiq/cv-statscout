import { useRef, useState } from "react";

const ACCEPTED_TYPES = ["video/mp4", "video/x-msvideo", "video/quicktime", "video/x-matroska"];
const ACCEPTED_EXT = ".mp4,.avi,.mov,.mkv";

export default function UploadZone({ onUpload }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = (file) => {
    if (!file) return;
    if (!ACCEPTED_TYPES.includes(file.type)) {
      alert(`Unsupported file type: ${file.type}\nPlease upload MP4, AVI, MOV, or MKV.`);
      return;
    }
    onUpload(file);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  };

  return (
    <div
      className={`upload-zone${dragOver ? " drag-over" : ""}`}
      onClick={() => inputRef.current.click()}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
    >
      <div className="upload-icon">🎬</div>
      <h2>Upload Match Video</h2>
      <p>Drag & drop a football match video here, or click to browse</p>
      <p>Supported: MP4, AVI, MOV, MKV &mdash; up to 2 GB</p>
      <span className="upload-btn">Choose file</span>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_EXT}
        style={{ display: "none" }}
        onChange={(e) => handleFile(e.target.files[0])}
      />
    </div>
  );
}
