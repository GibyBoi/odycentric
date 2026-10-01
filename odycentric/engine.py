"""Background removal: the model files, the cutout itself, and saving results.

No Qt in here. This runs inside the worker process, never in the app window.

The AI never redraws the photo. It only decides how see-through each pixel
should be. Wherever the subject is solid, the output pixel is copied from the
original byte for byte, so faces, text and logos come through unchanged.
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pillow_heif
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"

GB = 1024**3


@dataclass(frozen=True)
class Model:
    key: str
    label: str
    blurb: str
    file: str
    size: int
    sha256: str
    native: int  # the model only ever sees native x native pixels
    memory: int  # peak memory while it runs (measured for 1024, estimated for 2048)

    @property
    def path(self):
        return MODELS_DIR / self.file

    @property
    def url(self):
        return MODEL_URL + self.file

    def is_downloaded(self):
        return self.path.is_file() and self.path.stat().st_size == self.size

    def resolutions(self):
        """AI resolutions on offer: the model's own, then tiled detail passes up to 4096."""
        return [r for r in RESOLUTIONS if r >= self.native]


# BiRefNet (MIT licence), exported to ONNX by the rembg project.
MODELS = {m.key: m for m in (
    Model("people", "People",
          "Trained on portraits. Best for photos of people.",
          "BiRefNet-portrait-epoch_150.onnx", 972_666_916,
          "1ba1c8ff5a7bbfadc8d8d13fb11d7be793f91f23d9d466549e37a854f6668f99", 1024, 7 * GB),
    Model("anything", "Anything",
          "General purpose: products, pets, objects, scenes.",
          "BiRefNet-general-epoch_244.onnx", 972_666_916,
          "58f621f00f5d756097615970a88a791584600dcf7c45b18a0a6267535a1ebd3c", 1024, 7 * GB),
    Model("fast", "Fast",
          "Small general model. About twice as quick, slightly rougher edges.",
          "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx", 224_005_088,
          "5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333", 1024, 7 * GB),
    Model("people_hd", "People HD (matting)",
          "Sees the photo at 2048 px and is trained for soft hair edges. Needs about 22 GB of free memory.",
          "BiRefNet_HR-matting-epoch_135.onnx", 1_098_928_867,
          "45b7d92ce0e2e75e1c8a564a9e84c87ac9da7f4315d2deeef3674864e809aef3", 2048, 22 * GB),
    Model("anything_hd", "Anything HD",
          "General purpose at 2048 px. Needs about 22 GB of free memory.",
          "BiRefNet_HR-general-epoch_130.onnx", 1_098_928_953,
          "db0217e99b25e0c4f6f4dca2892ff1f7ea7aba38fb6ad84f93122a4024be536a", 2048, 22 * GB),
)}

RESOLUTIONS = (1024, 2048, 3072, 4096)
PHOTO_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
MAX_PIXELS = 100_000_000
OUTPUT_SUFFIX = "-nobg"

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
# Mask values this close to 0 or 255 snap to fully clear or fully solid, so the
# inside of the subject is exact original pixels rather than a near-copy.
SNAP = 4
# Edge colour cleanup is worked out at this many pixels at most, then scaled up.
# It only touches see-through pixels, and this keeps big photos fast and lean.
REFINE_PIXELS = 2_000_000
# Detail tiles only run where the first pass was unsure: probability in here.
UNSURE = (0.02, 0.98)
# Edge clarity: a see-through band this many pixels wide (on a 1024 px grid) or
# thinner scores 100. Twice as wide scores 50.
CLEAN_BAND = 2.0

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class PhotoError(Exception):
    """A problem with one photo, worded for the person using the app."""


def open_photo(path):
    """Returns the photo upright and in RGB, plus its colour profile."""
    try:
        image = Image.open(path)
    except Image.DecompressionBombError:
        raise PhotoError(f"Photo is larger than {MAX_PIXELS // 1_000_000} megapixels.")
    if image.width * image.height > MAX_PIXELS:
        raise PhotoError(f"Photo is larger than {MAX_PIXELS // 1_000_000} megapixels.")
    icc = image.info.get("icc_profile")
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image, icc


class Remover:
    def __init__(self, model_path, threads):
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads  # 0 lets ONNX Runtime pick
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Free each buffer as soon as it is done with instead of pooling them.
        # Pooling pushed peak memory for the 1024 models from 7 GB to 10 GB.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        # CPU only. See requirements.txt for why the GPU is never used.
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.native = self.session.get_inputs()[0].shape[-1]

    def _predict(self, image):
        """Subject probability (0 to 1) for an image exactly native x native."""
        x = np.asarray(image, np.float32) / 255
        x = np.ascontiguousarray(((x - MEAN) / STD).transpose(2, 0, 1)[None])
        logits = self.session.run(None, {self.input_name: x})[0][0, 0]
        return 1 / (1 + np.exp(-np.clip(logits, -30, 30)))

    def probability(self, photo, resolution):
        """Subject probability over the photo, and how many AI passes it took.

        One pass sees the whole photo squeezed to native x native. If the chosen
        resolution is higher, the photo is also scaled to that size and the AI
        looks again, tile by tile, but only along the outline where the first
        pass was unsure. The first pass keeps the big picture, so a tile that
        only shows background can never add stray objects far from the subject.
        """
        n = self.native
        whole = self._predict(photo.resize((n, n), Image.BILINEAR))
        w, h = photo.size
        scale = min(resolution, max(w, h)) / max(w, h)
        work_w, work_h = max(1, round(w * scale)), max(1, round(h * scale))
        if max(work_w, work_h) <= n:
            return whole, 1
        prob = cv2.resize(whole, (work_w, work_h), interpolation=cv2.INTER_LINEAR)
        unsure = (prob > UNSURE[0]) & (prob < UNSURE[1])
        if not unsure.any():
            return prob, 1
        # widen the band a little: the first pass may have put the edge slightly off
        reach = max(3, 2 * round(max(work_w, work_h) / n) + 1)
        unsure = cv2.dilate(unsure.astype(np.uint8), np.ones((reach, reach), np.uint8)).astype(bool)

        pad_w, pad_h = max(work_w, n), max(work_h, n)
        work = np.asarray(photo.resize((work_w, work_h), Image.BILINEAR))
        work = np.pad(work, ((0, pad_h - work_h), (0, pad_w - work_w), (0, 0)), mode="edge")
        unsure_padded = np.pad(unsure, ((0, pad_h - work_h), (0, pad_w - work_w)))
        step = n * 3 // 4
        ramp = np.minimum(1.0, np.minimum(np.arange(1, n + 1), np.arange(n, 0, -1)) / (n / 8))
        feather = np.outer(ramp, ramp).astype(np.float32)
        total = np.zeros((pad_h, pad_w), np.float32)
        weight = np.zeros((pad_h, pad_w), np.float32)
        passes = 1
        for y in _tile_starts(pad_h, n, step):
            for x in _tile_starts(pad_w, n, step):
                if not unsure_padded[y:y + n, x:x + n].any():
                    continue
                tile = self._predict(Image.fromarray(work[y:y + n, x:x + n]))
                total[y:y + n, x:x + n] += tile * feather
                weight[y:y + n, x:x + n] += feather
                passes += 1
        total, weight = total[:work_h, :work_w], weight[:work_h, :work_w]
        refined = unsure & (weight > 0)
        prob[refined] = total[refined] / weight[refined]
        return prob, passes


def _tile_starts(length, tile, step):
    starts = list(range(0, max(length - tile, 0) + 1, step))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def edge_clarity(prob, size):
    """0 to 100: how decisively the AI split subject from background.

    Measured on a 1024 px grid so every photo and setting is judged alike: the
    number of see-through pixels divided by the length of the subject's outline
    gives the average width of the see-through band. A crisp cut is a band about
    CLEAN_BAND pixels wide. Wide bands mean the AI was unsure (or the subject is
    genuinely wispy, which is what star ratings are for).
    """
    w, h = size
    grid = (max(1, round(1024 * w / max(w, h))), max(1, round(1024 * h / max(w, h))))
    p = cv2.resize(prob, grid, interpolation=cv2.INTER_AREA)
    solid = (p >= 0.5).astype(np.uint8)
    if solid.sum() < 100:
        return 0
    outline = int((solid - cv2.erode(solid, np.ones((3, 3), np.uint8))).sum())
    band = int(((p > 0.05) & (p < 0.95)).sum())
    width = band / max(outline, 1)
    return round(100 * min(1.0, CLEAN_BAND / max(width, 1e-6)))


def to_alpha(prob, size, sharp=False):
    """Probability map to a full-size 8-bit alpha channel."""
    alpha = cv2.resize(prob, size, interpolation=cv2.INTER_LINEAR)
    alpha = np.clip(np.round(alpha * 255), 0, 255).astype(np.uint8)
    if sharp:
        return np.where(alpha >= 128, np.uint8(255), np.uint8(0))
    alpha[alpha <= SNAP] = 0
    alpha[alpha >= 255 - SNAP] = 255
    return alpha


def compose(photo, alpha, background=None, cleanup=True):
    """Original pixels where the subject is solid, colour-cleaned pixels on soft
    edges like hair, and nothing at all where the background was. Zeroing the
    background matters: a PNG that only hides it still carries it."""
    colour = np.array(photo)
    soft = (alpha > 0) & (alpha < 255)
    if cleanup and soft.any():
        np.copyto(colour, _edge_colours(photo, alpha), where=soft[..., None])
    np.copyto(colour, np.uint8(0), where=(alpha == 0)[..., None])
    result = Image.fromarray(colour)
    result.putalpha(Image.fromarray(alpha))
    if background is not None:
        canvas = Image.new("RGBA", result.size, (*background, 255))
        result = Image.alpha_composite(canvas, result).convert("RGB")
    return result


def _edge_colours(photo, alpha):
    """Foreground colour on see-through pixels with the old background's tint
    taken out, so hair doesn't keep a halo of the original backdrop. This is the
    blur-fusion estimator (Forte and Pitie, 2021) that BiRefNet itself uses."""
    w, h = photo.size
    scale = min(1.0, math.sqrt(REFINE_PIXELS / (w * h)))
    if scale < 1:
        small = (max(1, round(w * scale)), max(1, round(h * scale)))
        photo = photo.resize(small, Image.BILINEAR)
        alpha = np.asarray(Image.fromarray(alpha).resize(small, Image.BILINEAR))
    image = np.asarray(photo, np.float32) / 255
    a = (alpha.astype(np.float32) / 255)[..., None]
    fg, bg = _blur_fusion(image, image, image, a, 90)
    fg, _ = _blur_fusion(image, fg, bg, a, 6)
    fg = Image.fromarray(np.round(fg * 255).astype(np.uint8))
    if scale < 1:
        fg = fg.resize((w, h), Image.BILINEAR)
    return np.asarray(fg)


def _blur_fusion(image, fg, bg, a, radius):
    k = (radius, radius)
    blurred_a = cv2.blur(a, k).reshape(a.shape)
    blurred_fg = cv2.blur(fg * a, k) / (blurred_a + 1e-5)
    blurred_bg = cv2.blur(bg * (1 - a), k) / ((1 - blurred_a) + 1e-5)
    fg = blurred_fg + a * (image - a * blurred_fg - (1 - a) * blurred_bg)
    return np.clip(fg, 0, 1), blurred_bg


def output_path(src, out_dir):
    """Never overwrites anything: photo.jpg -> photo-nobg.png, then photo-nobg (2).png."""
    out_dir, stem = Path(out_dir), Path(src).stem
    path, n = out_dir / f"{stem}{OUTPUT_SUFFIX}.png", 2
    while path.exists():
        path, n = out_dir / f"{stem}{OUTPUT_SUFFIX} ({n}).png", n + 1
    return path


def process(remover, src, out_dir, resolution=1024, max_side=None, sharp=False, cleanup=True, background=None):
    """Cut out one photo and save it. Returns where it went and how it went."""
    started = time.perf_counter()
    photo, icc = open_photo(src)
    megapixels = photo.width * photo.height / 1e6
    if max_side and max(photo.size) > max_side:
        scale = max_side / max(photo.size)
        photo = photo.resize((max(1, round(photo.width * scale)), max(1, round(photo.height * scale))),
                             Image.LANCZOS)
    ai_started = time.perf_counter()
    prob, passes = remover.probability(photo, resolution)
    ai_seconds = time.perf_counter() - ai_started
    clarity = edge_clarity(prob, photo.size)
    result = compose(photo, to_alpha(prob, photo.size, sharp), background, cleanup)
    dst = output_path(src, out_dir)
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_name(dst.name + ".part")
    result.save(partial, format="PNG", icc_profile=icc, compress_level=3)
    partial.replace(dst)
    return {"dst": str(dst), "seconds": round(time.perf_counter() - started, 2), "ai_seconds": round(ai_seconds, 2),
            "passes": passes, "clarity": clarity, "megapixels": round(megapixels, 2),
            "out_size": list(photo.size)}
