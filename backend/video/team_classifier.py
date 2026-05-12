import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import umap
from typing import Optional
import torch
from transformers import AutoImageProcessor, AutoModel


class TeamClassifier:
    """
    SigLIP embeddings + UMAP + KMeans for team-label clustering.

    Usage: collect player crops from many frames, call fit_crops() once,
    then call predict_crops() in the main processing loop.
    """

    SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
    EMBEDDING_DIM = 768
    UMAP_N_COMPONENTS = 3
    BATCH_SIZE = 32

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TeamClassifier] Loading SigLIP on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(self.SIGLIP_MODEL_ID)
        self.model = AutoModel.from_pretrained(self.SIGLIP_MODEL_ID).to(self.device)
        self.model.eval()
        self.kmeans: Optional[KMeans] = None
        self.reducer: Optional[umap.UMAP] = None
        self.fitted = False

    def _bgr_to_pil(self, crops_bgr: list[np.ndarray]) -> list[Image.Image]:
        return [Image.fromarray(c[:, :, ::-1]) for c in crops_bgr if c.size > 0]

    def _embed(self, pil_crops: list[Image.Image], verbose: bool = False) -> np.ndarray:
        if not pil_crops:
            return np.empty((0, self.EMBEDDING_DIM))
        all_embeddings = []
        n_batches = (len(pil_crops) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        for bi, i in enumerate(range(0, len(pil_crops), self.BATCH_SIZE)):
            batch = pil_crops[i : i + self.BATCH_SIZE]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            all_embeddings.append(outputs.cpu().numpy())
            if verbose:
                print(f"[TeamClassifier] Embedded batch {bi + 1}/{n_batches}")
        embeddings = np.concatenate(all_embeddings, axis=0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.where(norms == 0, 1, norms)

    def fit_crops(self, crops_bgr: list[np.ndarray]):
        """Fit on a large batch of player crops collected from across the video."""
        print(f"[TeamClassifier] Fitting on {len(crops_bgr)} crops...")
        pil_crops = self._bgr_to_pil(crops_bgr)
        embeddings = self._embed(pil_crops, verbose=True)
        if len(embeddings) < 4:
            print(f"[TeamClassifier] Not enough crops ({len(embeddings)}) to fit; skipping.")
            return
        print("[TeamClassifier] Running UMAP...")
        self.reducer = umap.UMAP(n_components=self.UMAP_N_COMPONENTS, random_state=42)
        reduced = self.reducer.fit_transform(embeddings)
        print("[TeamClassifier] Running KMeans...")
        self.kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.kmeans.fit(reduced)
        self.fitted = True
        print(f"[TeamClassifier] Fit complete on {len(embeddings)} crops.")

    def predict_crops(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        if not self.fitted or len(crops_bgr) == 0:
            return np.zeros(len(crops_bgr), dtype=int)
        pil_crops = self._bgr_to_pil(crops_bgr)
        embeddings = self._embed(pil_crops)
        if len(embeddings) == 0:
            return np.zeros(len(crops_bgr), dtype=int)
        reduced = self.reducer.transform(embeddings)
        return self.kmeans.predict(reduced)
