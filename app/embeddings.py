# embeddings.py
import io
import zlib
import numpy as np
from PIL import Image
import tempfile
import os

# Lazy model holder
_deepface_model = None
_deepface_backend_name = "ArcFace"  # DeepFace backend

# CUSTOM_TEMP = "D:\\PROJECTS\\Face Recognition\\image_uploads"
# os.makedirs(CUSTOM_TEMP, exist_ok=True)

def _init_deepface():
    """
    Initialize DeepFace ArcFace model.
    """
    global _deepface_model
    if _deepface_model is None:
        try:
            from deepface import DeepFace
        except Exception as e:
            raise RuntimeError(
                "DeepFace is not installed. Install with `pip install deepface`."
            ) from e

        # Build model (DeepFace will download weights on first run)
        model = DeepFace.build_model(_deepface_backend_name)
        _deepface_model = model
    return _deepface_model


def get_embedding_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Convert image bytes to a normalized ArcFace embedding (float32) using DeepFace.
    Tries numpy array directly (fast). Falls back to tempfile (safe).
    """
    from deepface import DeepFace
    import tempfile
    import os

    # load image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    rep = None
    try:
        # 🚀 Fast path: numpy array
        rep = DeepFace.represent(
            img_path=np.array(img),
            model_name=_deepface_backend_name,
            enforce_detection=True
        )
    except Exception as e:
        print(f"[WARN] Numpy input failed, falling back to temp file: {e}")

        # custom temp folder
        CUSTOM_TEMP = "D:\\PROJECTS\\Face Recognition\\image_uploads"
        os.makedirs(CUSTOM_TEMP, exist_ok=True)

        # Safe tempfile for Windows
        fd, path = tempfile.mkstemp(suffix=".jpg", dir=CUSTOM_TEMP)
        os.close(fd)
        try:
            img.save(path, format="JPEG")
            rep = DeepFace.represent(
                img_path=path,
                model_name=_deepface_backend_name,
                enforce_detection=True
            )
        finally:
            if os.path.exists(path):
                os.remove(path)

    # parse embedding
    emb = None
    if isinstance(rep, list) and len(rep) > 0:
        first = rep[0]
        if isinstance(first, dict) and "embedding" in first:
            emb = np.array(first["embedding"], dtype=np.float32)
    elif isinstance(rep, dict) and "embedding" in rep:
        emb = np.array(rep["embedding"], dtype=np.float32) # type: ignore

    if emb is None:
        raise RuntimeError("DeepFace did not return an embedding.")

    # normalize
    norm = np.linalg.norm(emb)
    if norm == 0:
        raise RuntimeError("Zero-norm embedding from DeepFace.")
    return (emb / norm).astype(np.float32)


def emb_to_bytes(emb: np.ndarray) -> bytes:
    """Compress and convert embedding to bytes for DB storage (float16 + zlib)."""
    f16 = emb.astype(np.float16)
    raw = f16.tobytes()
    compressed = zlib.compress(raw, level=6)
    return compressed


def bytes_to_emb(blob: bytes) -> np.ndarray:
    """Decompress bytes back to float32 numpy array."""
    raw = zlib.decompress(blob)
    arr = np.frombuffer(raw, dtype=np.float16)
    return arr.astype(np.float32)
