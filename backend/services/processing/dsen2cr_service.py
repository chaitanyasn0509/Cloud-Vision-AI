"""GPU inference adapter for the pretrained DSen2-CR CARL checkpoint."""
from __future__ import annotations

import math
import subprocess
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import rasterio
import torch
import torch.nn as nn
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile

from visualization import to_rgb, sar_preview
from .inference_service import InferenceDiagnostics, InferenceResult, S2_BAND_NAMES, get_device


class _ResidualBlock(nn.Module):
    def __init__(self, features: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.conv2(self.relu(self.conv1(x)))


class _DSen2CRNetwork(nn.Module):
    """PyTorch equivalent of the bundled 16-block DSen2-CR network."""
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(15, 256, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([_ResidualBlock(256) for _ in range(16)])
        self.conv_out = nn.Conv2d(256, 13, kernel_size=3, padding=1, bias=True)

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        features = self.relu(self.conv_in(torch.cat((optical, sar), dim=1)))
        for block in self.blocks:
            features = block(features)
        return optical + self.conv_out(features)


class DSen2CRResult(InferenceResult):
    """InferenceResult carrying model-specific DSen2-CR export metadata."""

    def cloud_guided_prediction(self, threshold: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
        """Blend the CNN result into cloudy areas while retaining clear input pixels."""
        score = getattr(self, "cloud_score", None)
        model_prediction = getattr(self, "model_prediction", self.prediction)
        cloudy_reflectance = getattr(self, "cloudy_reflectance", None)
        if score is None or cloudy_reflectance is None:
            raise ValueError("Cloud-score inputs are unavailable for this CNN result.")
        strength = np.clip((score - (threshold - 0.06)) / 0.12, 0.0, 1.0)
        blended = cloudy_reflectance * (1.0 - strength[None, :, :]) + model_prediction * strength[None, :, :]
        blended[:, ~self.valid_mask] = 0.0
        return np.clip(blended, 0.0, 1.0).astype(np.float32), strength

    def to_diagnostics_export(self, image_id: str):
        export = super().to_diagnostics_export(image_id)
        export["stage_name"] = "DSen2-CR Cloud Reconstruction"
        export["input_quality"] = getattr(self, "input_quality", {})
        score = getattr(self, "cloud_score", None)
        if score is not None:
            valid = np.asarray(self.valid_mask, dtype=bool)
            export["cloud_guided_output"] = {
                "mode": getattr(self, "output_mode", "Raw DSen2-CR"),
                "score_threshold": getattr(self, "cloud_score_threshold", None),
                "mean_cloud_score": float(np.mean(score[valid])) if np.any(valid) else 0.0,
            }
        return export

    def save_geotiff(self, path: Union[str, Path]) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        profile = self.profile.copy()
        profile.update(count=13, dtype="float32", nodata=0.0, compress="deflate")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(self.prediction.astype(np.float32))
            for index, name in enumerate(S2_BAND_NAMES, start=1):
                dst.set_band_description(index, name)
            dst.update_tags(processing="DSen2-CR cloud reconstruction", model="DSen2-CR (CARL)",
                           device=self.diagnostics.device,
                           execution_time_s=str(round(self.diagnostics.execution_time_seconds, 2)))
        self.output_tif_path = out_path
        return out_path

    def to_geotiff_bytes(self) -> bytes:
        profile = self.profile.copy()
        profile.update(count=13, dtype="float32", nodata=0.0, compress="deflate")
        with MemoryFile() as memory:
            with memory.open(**profile) as dst:
                dst.write(self.prediction.astype(np.float32))
                for index, name in enumerate(S2_BAND_NAMES, start=1):
                    dst.set_band_description(index, name)
                dst.update_tags(processing="DSen2-CR cloud reconstruction", model="DSen2-CR (CARL)")
            return memory.read()

    def to_rgb_geotiff_bytes(self) -> bytes:
        rgb = np.rint(np.clip(self.prediction[[3, 2, 1]], 0.0, 1.0) * 10000.0).astype(np.uint16)
        mask = np.asarray(self.valid_mask).squeeze().astype(bool)
        if mask.shape == rgb.shape[1:]:
            rgb[:, ~mask] = 0
        profile = self.profile.copy()
        profile.update(count=3, dtype="uint16", nodata=0, compress="deflate", predictor=2)
        with MemoryFile() as memory:
            with memory.open(**profile) as dst:
                dst.write(rgb)
                for index, name in enumerate(("B04", "B03", "B02"), start=1):
                    dst.set_band_description(index, name)
                dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
                dst.update_tags(processing="DSen2-CR RGB export", model="DSen2-CR (CARL)", bands="B04,B03,B02")
            return memory.read()


class DSen2CRService:
    TILE_SIZE = 128
    OVERLAP = 32

    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        self.root = Path(project_root) if project_root else Path(__file__).resolve().parents[3]
        self.model_root = self.root / "dsen2-cr-main"
        self.checkpoint = self.model_root / "checkpoints" / "model_SARcarl.hdf5"
        self.converter = self.model_root / "Code" / "export_torch_weights.py"
        self.converter_python = self.model_root / ".venv37" / "Scripts" / "python.exe"
        self.torch_weights = self.model_root / "checkpoints" / "model_SARcarl.torch.npz"
        self.device = get_device()
        self._model: Optional[_DSen2CRNetwork] = None

    @property
    def available(self) -> bool:
        return self.checkpoint.is_file() and self.converter.is_file()

    def _ensure_torch_weights(self) -> None:
        if self.torch_weights.exists() and self.torch_weights.stat().st_mtime >= self.checkpoint.stat().st_mtime:
            return
        if not self.converter_python.is_file():
            raise FileNotFoundError("The bundled Python 3.7 + h5py environment is needed once to convert the DSen2-CR HDF5 weights.")
        completed = subprocess.run(
            [str(self.converter_python), str(self.converter), str(self.checkpoint), str(self.torch_weights)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if completed.returncode != 0 or not self.torch_weights.is_file():
            details = (completed.stderr or completed.stdout or "No conversion details returned.").strip()
            raise RuntimeError("Could not convert DSen2-CR checkpoint for PyTorch:\n" + details[-6000:])

    @staticmethod
    def _positions(length: int, tile: int, overlap: int) -> list[int]:
        step = tile - overlap
        positions = list(range(0, max(1, length - tile + 1), step))
        if not positions or positions[-1] + tile < length:
            positions.append(max(0, length - tile))
        return sorted(set(positions))

    def _get_model(self) -> _DSen2CRNetwork:
        if self._model is not None:
            return self._model
        self._ensure_torch_weights()
        model = _DSen2CRNetwork()
        with np.load(self.torch_weights, allow_pickle=False) as weights:
            def copy_conv(layer: nn.Conv2d, name: str) -> None:
                kernel = weights[name + ".kernel"].transpose(3, 2, 0, 1).copy()
                bias = weights[name + ".bias"].copy()
                layer.weight.data.copy_(torch.from_numpy(kernel))
                layer.bias.data.copy_(torch.from_numpy(bias))

            copy_conv(model.conv_in, "conv2d_1")
            for block_index, block in enumerate(model.blocks):
                first_layer = 2 + block_index * 2
                copy_conv(block.conv1, "conv2d_{}".format(first_layer))
                copy_conv(block.conv2, "conv2d_{}".format(first_layer + 1))
            copy_conv(model.conv_out, "conv2d_34")
        model.to(self.device)
        model.eval()
        self._model = model
        return model

    @staticmethod
    def _normalize_s1(s1: np.ndarray) -> np.ndarray:
        normalized = np.nan_to_num(s1.astype(np.float32), nan=-25.0, posinf=0.0, neginf=-32.5)
        for band, low in enumerate((-25.0, -32.5)):
            normalized[band] = 2.0 * (np.clip(normalized[band], low, 0.0) - low) / (0.0 - low)
        return normalized

    @staticmethod
    def _input_quality_report(s1: np.ndarray, s2: np.ndarray, valid_mask: np.ndarray) -> dict:
        report = {
            "sar_expected_db": {"VV": [-25.0, 0.0], "VH": [-32.5, 0.0]},
            "sar_bands": {},
            "optical_expected_dn": [0.0, 10000.0],
            "optical_bands": {},
        }
        for index, band in enumerate(("VV", "VH")):
            values = s1[index][valid_mask & np.isfinite(s1[index])]
            low = -25.0 if index == 0 else -32.5
            if values.size:
                report["sar_bands"][band] = {
                    "p02_db": float(np.percentile(values, 2)),
                    "p50_db": float(np.percentile(values, 50)),
                    "p98_db": float(np.percentile(values, 98)),
                    "below_training_range_fraction": float(np.mean(values < low)),
                    "above_training_range_fraction": float(np.mean(values > 0.0)),
                }
        for index, band in enumerate(S2_BAND_NAMES):
            values = s2[index][valid_mask & np.isfinite(s2[index])]
            if values.size:
                report["optical_bands"][band] = {
                    "p02_dn": float(np.percentile(values, 2)),
                    "p50_dn": float(np.percentile(values, 50)),
                    "p98_dn": float(np.percentile(values, 98)),
                    "below_training_range_fraction": float(np.mean(values < 0.0)),
                    "above_training_range_fraction": float(np.mean(values > 10000.0)),
                }
        vv = report["sar_bands"].get("VV", {})
        report["warnings"] = []
        if vv.get("above_training_range_fraction", 0.0) > 0.05:
            report["warnings"].append(
                "More than 5% of VV input exceeds the model's 0 dB training limit and will be clipped."
            )
        if vv.get("below_training_range_fraction", 0.0) > 0.05:
            report["warnings"].append(
                "More than 5% of VV input is below the model's -25 dB training limit and will be clipped."
            )
        vh = report["sar_bands"].get("VH", {})
        if vh.get("above_training_range_fraction", 0.0) > 0.05:
            report["warnings"].append(
                "More than 5% of VH input exceeds the model's 0 dB training limit and will be clipped."
            )
        if vh.get("below_training_range_fraction", 0.0) > 0.05:
            report["warnings"].append(
                "More than 5% of VH input is below the model's -32.5 dB training limit and will be clipped."
            )
        return report

    @staticmethod
    def _cloud_score(s2: np.ndarray) -> np.ndarray:
        """SEN12MS-CR's spectral cloud score used by the original DSen2-CR loader."""
        from scipy import ndimage, signal

        reflectance = np.nan_to_num(s2.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0) / 10000.0

        def rescale(values: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
            return (values - limits[0]) / (limits[1] - limits[0])

        score = np.ones(reflectance.shape[1:], dtype=np.float32)
        score = np.minimum(score, rescale(reflectance[1], (0.1, 0.5)))
        score = np.minimum(score, rescale(reflectance[0], (0.1, 0.3)))
        score = np.minimum(score, rescale(reflectance[0] + reflectance[10], (0.15, 0.2)))
        score = np.minimum(score, rescale(reflectance[3] + reflectance[2] + reflectance[1], (0.2, 0.8)))
        ndsi_denominator = reflectance[2] + reflectance[11]
        ndsi_denominator[ndsi_denominator == 0] = 0.001
        ndsi = (reflectance[2] - reflectance[11]) / ndsi_denominator
        score = np.minimum(score, rescale(ndsi, (0.8, 0.6)))
        score = ndimage.grey_closing(score, size=(5, 5))
        score = signal.convolve2d(score, np.ones((7, 7), dtype=np.float32) / 49.0, mode="same")
        return np.clip(score, 0.00001, 1.0).astype(np.float32)

    def run(self, s1_input: Union[str, Path], s2_cloudy_input: Union[str, Path],
            validity_mask_input: Optional[Union[str, Path]] = None,
            brightness: float = 2.5,
            progress_callback=None) -> DSen2CRResult:
        if not self.available:
            raise FileNotFoundError("DSen2-CR CARL checkpoint or converter is missing from dsen2-cr-main/.")
        started = time.perf_counter()
        with rasterio.open(s2_cloudy_input) as s2_src, rasterio.open(s1_input) as s1_src:
            if s2_src.count != 13 or s1_src.count != 2:
                raise ValueError("DSen2-CR requires harmonized 13-band S2 and 2-band S1 GeoTIFFs.")
            if (s2_src.width, s2_src.height) != (s1_src.width, s1_src.height) or s2_src.crs != s1_src.crs:
                raise ValueError("Sentinel-1 and Sentinel-2 inputs must be spatially harmonized first.")
            s2 = s2_src.read().astype(np.float32)
            s1 = s1_src.read().astype(np.float32)
            profile = s2_src.profile.copy()
            transform = s2_src.transform
            crs = s2_src.crs
        height, width = s2.shape[1:]
        if validity_mask_input:
            with rasterio.open(validity_mask_input) as mask_src:
                valid_mask = mask_src.read(1).astype(bool)
        else:
            valid_mask = np.isfinite(s2[0]) & (s2[0] > 0.0) & np.isfinite(s1[0]) & (s1[0] > -500.0)
        if valid_mask.shape != (height, width):
            raise ValueError("Validity mask dimensions do not match the Sentinel-2 input.")

        input_quality = self._input_quality_report(s1, s2, valid_mask)
        cloud_score = self._cloud_score(s2)
        cloud_score[~valid_mask] = 0.0

        optical = np.nan_to_num(s2, nan=0.0, posinf=0.0, neginf=0.0)
        optical = np.clip(optical, 0.0, 10000.0) / 2000.0
        sar = self._normalize_s1(s1)
        pad_h = max(0, self.TILE_SIZE - height)
        pad_w = max(0, self.TILE_SIZE - width)
        if pad_h or pad_w:
            mode = "reflect" if height > 1 and width > 1 else "edge"
            optical = np.pad(optical, ((0, 0), (0, pad_h), (0, pad_w)), mode=mode)
            sar = np.pad(sar, ((0, 0), (0, pad_h), (0, pad_w)), mode=mode)

        padded_height, padded_width = optical.shape[1:]
        xs = self._positions(padded_width, self.TILE_SIZE, self.OVERLAP)
        ys = self._positions(padded_height, self.TILE_SIZE, self.OVERLAP)
        total_tiles = len(xs) * len(ys)
        model = self._get_model()
        prediction_sum = np.zeros((13, padded_height, padded_width), dtype=np.float32)
        weights = np.zeros((padded_height, padded_width), dtype=np.float32)
        completed_tiles = 0
        with torch.inference_mode():
            for y in ys:
                for x in xs:
                    optical_tile = torch.from_numpy(optical[:, y:y + self.TILE_SIZE, x:x + self.TILE_SIZE].copy()).unsqueeze(0).to(self.device)
                    sar_tile = torch.from_numpy(sar[:, y:y + self.TILE_SIZE, x:x + self.TILE_SIZE].copy()).unsqueeze(0).to(self.device)
                    output = model(optical_tile, sar_tile).squeeze(0).detach().cpu().numpy()
                    prediction_sum[:, y:y + self.TILE_SIZE, x:x + self.TILE_SIZE] += output
                    weights[y:y + self.TILE_SIZE, x:x + self.TILE_SIZE] += 1.0
                    completed_tiles += 1
                    if progress_callback is not None:
                        progress_callback(completed_tiles, total_tiles)

        # DSen2-CR is trained in DN / 2000 units, whose full scale is 5.
        prediction = prediction_sum / np.maximum(weights, 1e-6)[None, :, :]
        prediction = np.clip(prediction[:, :height, :width] / 5.0, 0.0, 1.0)
        prediction[:, ~valid_mask] = 0.0
        prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        elapsed = time.perf_counter() - started
        valid_values = prediction[:, valid_mask]
        per_band_stats = {}
        for index, name in enumerate(S2_BAND_NAMES):
            values = prediction[index][valid_mask]
            per_band_stats[name] = {
                "min": float(values.min()) if values.size else 0.0,
                "max": float(values.max()) if values.size else 0.0,
                "mean": float(values.mean()) if values.size else 0.0,
            }
        device_name = str(self.device)
        if self.device.type == "cuda":
            device_name = "CUDA ({}; DSen2-CR PyTorch)".format(torch.cuda.get_device_name(self.device))
        diagnostics = InferenceDiagnostics(
            device=device_name,
            execution_time_seconds=elapsed,
            tile_count=total_tiles,
            total_pixels=width * height,
            valid_pixels=int(valid_mask.sum()),
            invalid_pixels=int(width * height - valid_mask.sum()),
            output_shape=prediction.shape,
            min_val=float(valid_values.min()) if valid_values.size else 0.0,
            max_val=float(valid_values.max()) if valid_values.size else 0.0,
            mean_val=float(valid_values.mean()) if valid_values.size else 0.0,
            crs=str(crs) if crs else "",
            pixel_size_m=float(abs(transform.a)),
            per_band_stats=per_band_stats,
            nan_count=int(np.isnan(prediction).sum()),
            inf_count=int(np.isinf(prediction).sum()),
        )
        profile.update(count=13, dtype="float32", nodata=0.0, compress="deflate")
        result = DSen2CRResult(
            prediction=prediction,
            valid_mask=valid_mask,
            profile=profile,
            diagnostics=diagnostics,
            cloudy_rgb=to_rgb(s2, brightness=brightness, valid_mask=valid_mask),
            prediction_rgb=to_rgb(prediction, brightness=brightness, valid_mask=valid_mask),
            sar_preview_img=sar_preview(s1, valid_mask=valid_mask),
        )
        result.input_quality = input_quality
        result.cloud_score = cloud_score
        result.model_prediction = prediction.copy()
        result.cloudy_reflectance = np.clip(np.nan_to_num(s2, nan=0.0), 0.0, 10000.0) / 10000.0
        return result
