import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .depth import DepthEstimator
from .model import WallProjectionNet
from .postprocess import PRESETS, extract_rooms


MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
COLORS = np.array(
    [
        (61, 184, 255),
        (255, 99, 132),
        (88, 214, 141),
        (255, 194, 71),
        (171, 112, 255),
        (255, 126, 69),
        (56, 220, 210),
        (235, 92, 220),
    ],
    np.uint8,
)


@dataclass
class ExtractionResult:
    data: dict
    images: dict[str, np.ndarray]

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "rooms.json").write_text(
            json.dumps(self.data, indent=2), encoding="utf-8"
        )
        for name, image in self.images.items():
            Image.fromarray(image).save(folder / name)

    def archive(self):
        from zipfile import ZIP_DEFLATED, ZipFile

        # Build the API response entirely in memory.
        buffer = BytesIO()
        with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
            archive.writestr("rooms.json", json.dumps(self.data, indent=2))
            for name, image in self.images.items():
                encoded = BytesIO()
                Image.fromarray(image).save(encoded, format="PNG")
                archive.writestr(name, encoded.getvalue())
        return buffer.getvalue()


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return torch.device(name)


def letterbox(image, size=512):
    width, height = image.size
    scale = min(size / width, size / height)
    resized = image.resize(
        (round(width * scale), round(height * scale)), Image.Resampling.LANCZOS
    )
    offset = ((size - resized.width) // 2, (size - resized.height) // 2)
    canvas = Image.new("RGB", (size, size), (240, 240, 240))
    canvas.paste(resized, offset)

    # Preserve every transform parameter needed to map polygons back.
    return np.asarray(canvas), (
        width,
        height,
        scale,
        *offset,
        resized.width,
        resized.height,
    )


def restore(array, transform, interpolation):
    width, height, _, left, top, resized_width, resized_height = transform

    # Remove network padding before restoring the original resolution.
    cropped = array[top : top + resized_height, left : left + resized_width]
    return cv2.resize(cropped, (width, height), interpolation=interpolation)


def heatmap(probability):
    colored = cv2.applyColorMap(
        np.uint8(np.clip(probability, 0, 1) * 255), cv2.COLORMAP_TURBO
    )
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def annotate(rgb, rooms):
    image, layer = rgb.copy(), rgb.copy()

    # Draw fills and boundaries separately to keep both visible.
    for index, room in enumerate(rooms):
        polygon = np.asarray(room["polygon"], np.int32)
        color = tuple(int(value) for value in COLORS[index % len(COLORS)])
        cv2.fillPoly(layer, [polygon], color)
        cv2.polylines(image, [polygon], True, color, 3, cv2.LINE_AA)
    image = cv2.addWeighted(image, 0.55, layer, 0.45, 0)

    # A dark outline keeps area labels readable on bright plans.
    for room in rooms:
        polygon = np.asarray(room["polygon"], np.int32)
        center = tuple(np.round(polygon.mean(0)).astype(int))
        text = f"{room['id']}: {room['relative_area']:.1%}"
        cv2.putText(
            image,
            text,
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image


class FloorPlanExtractor:
    def __init__(self, weights=None, device="auto", depth_model=None):
        self.device = resolve_device(device)
        default = Path(__file__).resolve().parents[1] / "weights" / "wall_projection.pt"
        state = torch.load(
            weights or default, map_location=self.device, weights_only=True
        )
        self.size = int(state.get("size", 512))

        # Construct the exact checkpoint architecture without extra downloads.
        self.model = (
            WallProjectionNet(state.get("in_channels", 4)).to(self.device).eval()
        )
        self.model.load_state_dict(state["model"])
        self.depth = (
            DepthEstimator(self.device, depth_model)
            if depth_model
            else DepthEstimator(self.device)
        )

    @torch.inference_mode()
    def extract(self, source, preset="weak"):
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}")

        image = (
            Image.open(source).convert("RGB")
            if not isinstance(source, Image.Image)
            else source.convert("RGB")
        )
        rgb, transform = letterbox(image, self.size)
        depth = self.depth(rgb)
        rgb_tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
        depth_tensor = torch.from_numpy(depth)[None].float() * 2 - 1
        inputs = torch.cat(((rgb_tensor - MEAN) / STD, depth_tensor))[None].to(
            self.device
        )

        # Mixed precision reduces GPU memory while outputs remain float32.
        with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            probability = self.model(inputs)["bottom"][0, 0].float().cpu().numpy()

        walls, _, regions = extract_rooms(probability, preset)
        width, height, scale, left, top, _, _ = transform
        rooms = []

        # Convert polygons from the padded network canvas to source pixels.
        for region in regions:
            polygon = region.polygon.astype(np.float32)
            polygon[:, 0] = np.clip((polygon[:, 0] - left) / scale, 0, width - 1)
            polygon[:, 1] = np.clip((polygon[:, 1] - top) / scale, 0, height - 1)
            polygon = np.round(polygon).astype(int)
            area = float(abs(cv2.contourArea(polygon.astype(np.float32))))
            if area:
                rooms.append(
                    {
                        "id": len(rooms) + 1,
                        "polygon": polygon.tolist(),
                        "area_pixels": round(area, 1),
                    }
                )

        total = sum(room["area_pixels"] for room in rooms)
        for room in rooms:
            room["relative_area"] = room["area_pixels"] / total if total else 0.0
            room["image_area_fraction"] = room["area_pixels"] / (width * height)

        # Restore raster diagnostics to the same coordinates as the input.
        original = np.asarray(image)
        probability = restore(probability, transform, cv2.INTER_LINEAR)
        barrier = restore(walls.astype(np.uint8), transform, cv2.INTER_NEAREST).astype(
            bool
        )
        data = {
            "image_size": [width, height],
            "preset": preset,
            "room_count": len(rooms),
            "rooms": rooms,
        }
        images = {
            "annotated.png": annotate(original, rooms),
            "probability.png": heatmap(probability),
            "wall_barrier.png": np.repeat(barrier[..., None], 3, 2).astype(np.uint8)
            * 255,
        }
        return ExtractionResult(data, images)
