import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator:
    def __init__(self, device, model_id=MODEL_ID):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
        )

    @torch.inference_mode()
    def __call__(self, rgb):
        image = Image.fromarray(rgb)
        inputs = {
            key: value.to(self.device)
            for key, value in self.processor(images=image, return_tensors="pt").items()
        }
        with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            depth = self.model(**inputs).predicted_depth

        # Restore input resolution before computing robust scene-wise scaling.
        depth = F.interpolate(
            depth[:, None],
            rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        )[0, 0].float()

        # Percentiles suppress isolated depth outliers without assuming meters.
        low, high = torch.quantile(depth, 0.01), torch.quantile(depth, 0.99)
        normalized = ((depth - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
        return normalized.cpu().numpy()
