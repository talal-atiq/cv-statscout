const BASE_URL = "http://localhost:8000/api/video";

export async function uploadVideo(file, onProgress) {
  const formData = new FormData();
  formData.append("file", file);

  const xhr = new XMLHttpRequest();
  return new Promise((resolve, reject) => {
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = () => {
      if (xhr.status === 200) resolve(JSON.parse(xhr.responseText));
      else reject(new Error(`Upload failed: ${xhr.statusText}`));
    };
    xhr.onerror = () => reject(new Error("Network error during upload"));
    xhr.open("POST", `${BASE_URL}/upload`);
    xhr.send(formData);
  });
}

export async function getJobStatus(jobId) {
  const res = await fetch(`${BASE_URL}/status/${jobId}`);
  if (!res.ok) throw new Error("Status fetch failed");
  return res.json();
}
