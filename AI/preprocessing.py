"""Image preprocessing for chest X-ray analysis."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import pydicom
    from pydicom.multival import MultiValue

    HAS_PYDICOM = True
except ImportError:
    pydicom = None  # type: ignore[assignment]
    MultiValue = tuple  # type: ignore[misc, assignment]
    HAS_PYDICOM = False

_DICOM_EXTS = {".dcm", ".dicom"}


def _first_float(value, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        if HAS_PYDICOM and isinstance(value, MultiValue):
            return float(value[0])
        if isinstance(value, (list, tuple)):
            return float(value[0])
        return float(value)
    except (TypeError, ValueError, IndexError):
        return default


def _looks_like_dicom(path: Path) -> bool:
    if path.suffix.lower() in _DICOM_EXTS:
        return True
    try:
        with path.open("rb") as handle:
            header = handle.read(132)
        return len(header) >= 132 and header[128:132] == b"DICM"
    except OSError:
        return False


def _select_frame(arr: np.ndarray, ds) -> np.ndarray:
    """Keep a 2D grayscale or 3D color frame; drop extra multi-frame slices."""
    samples = int(_first_float(getattr(ds, "SamplesPerPixel", 1), 1) or 1)
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    n_frames = int(_first_float(getattr(ds, "NumberOfFrames", 1), 1) or 1)
    color = samples > 1 or photometric in {
        "RGB",
        "YBR_FULL",
        "YBR_FULL_422",
        "YBR_ICT",
        "YBR_RCT",
        "PALETTE COLOR",
    }

    if arr.ndim == 4:
        return arr[0]
    if arr.ndim != 3:
        return arr
    if color and arr.shape[-1] in (3, 4):
        return arr
    if color and arr.shape[0] in (3, 4) and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
        return np.moveaxis(arr, 0, -1)
    if n_frames > 1 or (not color and samples == 1):
        return arr[0]
    if arr.shape[-1] in (3, 4):
        return arr
    return arr[0]


def _to_grayscale_uint8(arr: np.ndarray, photometric: str) -> np.ndarray:
    if arr.ndim == 3:
        channels_last = arr.shape[-1] in (3, 4)
        work = arr if channels_last else np.moveaxis(arr, 0, -1)
        scaled = work if work.dtype == np.uint8 else normalize_to_uint8(work)
        if photometric.startswith("YBR"):
            gray = cv2.cvtColor(scaled, cv2.COLOR_YCrCb2GRAY)
        elif scaled.shape[-1] == 4:
            gray = cv2.cvtColor(scaled, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(scaled, cv2.COLOR_RGB2GRAY)
        return gray.astype(np.uint8)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])
    return arr.astype(np.uint8) if arr.dtype == np.uint8 else normalize_to_uint8(arr)


def extract_dicom_pixels(ds) -> np.ndarray:
    """Safely extract a displayable grayscale uint8 array from a pydicom dataset."""
    try:
        arr = ds.pixel_array
    except Exception as exc:
        raise ValueError("Unsupported or corrupted DICOM pixel data") from exc

    arr = _select_frame(np.asarray(arr), ds)

    slope = _first_float(getattr(ds, "RescaleSlope", 1.0), 1.0) or 1.0
    intercept = _first_float(getattr(ds, "RescaleIntercept", 0.0), 0.0) or 0.0
    try:
        from pydicom.pixel_data_handlers.util import apply_modality_lut

        arr = np.asarray(apply_modality_lut(arr, ds), dtype=np.float32)
    except Exception:
        arr = arr.astype(np.float32) * slope + intercept

    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper()

    try:
        from pydicom.pixel_data_handlers.util import apply_voi_lut

        if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
            arr = np.asarray(apply_voi_lut(arr, ds), dtype=np.float32)
        else:
            arr = normalize_to_uint8(arr).astype(np.float32)
    except Exception:
        center = _first_float(getattr(ds, "WindowCenter", None))
        width = _first_float(getattr(ds, "WindowWidth", None))
        if center is not None and width is not None and width > 0:
            arr = apply_windowing(arr, center, width).astype(np.float32)
        else:
            arr = normalize_to_uint8(arr).astype(np.float32)

    if photometric == "MONOCHROME1":
        arr = float(np.max(arr)) - arr

    return _to_grayscale_uint8(arr, photometric)


def load_image(image_path: str) -> np.ndarray:
    """Load PNG/JPG/DICOM and return grayscale uint8 array."""
    path = Path(image_path)
    try:
        if HAS_PYDICOM and _looks_like_dicom(path):
            ds = pydicom.dcmread(str(path), force=True)
            if not hasattr(ds, "PixelData"):
                raise ValueError("DICOM file has no pixel data")
            return extract_dicom_pixels(ds)

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            pil_img = Image.open(path).convert("L")
            img = np.array(pil_img)
        if img is None or img.size == 0:
            raise ValueError("Empty or unreadable image")
        return img
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Unsupported or corrupted image file") from exc


def apply_windowing(pixel_array: np.ndarray, center: float, width: float) -> np.ndarray:
    lower = center - width / 2
    upper = center + width / 2
    windowed = np.clip(pixel_array, lower, upper)
    return normalize_to_uint8(windowed)


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    return (arr * 255).astype(np.uint8)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)


def normalize_xrv(img: np.ndarray) -> np.ndarray:
    """Scale grayscale image to [-1024, 1024] 1-channel (1, H, W) float32 for TorchXRayVision."""
    img_f = img.astype(np.float32)
    img_min, img_max = float(img_f.min()), float(img_f.max())
    if img_max > img_min:
        scaled = (img_f - img_min) / (img_max - img_min) * 255.0
    else:
        scaled = np.zeros_like(img_f)
    norm = (2.0 * (scaled / 255.0) - 1.0) * 1024.0
    return norm[np.newaxis, :, :].astype(np.float32)


def preprocess_image(image_path: str, size: int = 224) -> np.ndarray:
    """
    Full preprocessing pipeline for TorchXRayVision DenseNet121:
    load -> CLAHE -> denoise -> resize -> normalize to [-1024, 1024] float32 1xHxW
    """
    try:
        gray = load_image(image_path)
        enhanced = apply_clahe(gray)
        cleaned = denoise(enhanced)
        resized = cv2.resize(cleaned, (size, size), interpolation=cv2.INTER_AREA)
        return normalize_xrv(resized)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Image preprocessing failed for unsupported or corrupted file") from exc
