"""Run LDSR-S2 on one Sentinel-2 RGBN GeoTIFF.

Drag a TIFF onto ``Run Super Resolution.bat`` or run this file with the TIFF
path as its only argument. The model requires real B4, B3, B2, and B8 data.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch

import opensr_utils

# opensr-utils writes Unicode status markers; Windows consoles may otherwise use cp1252.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
REQUIRED_BANDS = ("B4", "B3", "B2", "B8")


def choose_input() -> Path | None:
    """Show a file picker when the launcher is opened without a dropped file."""
    try:
        from tkinter import Tk, filedialog

        window = Tk()
        window.withdraw()
        window.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Choose a Sentinel-2 RGBN GeoTIFF",
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")],
        )
        window.destroy()
        return Path(selected) if selected else None
    except Exception as error:
        raise RuntimeError("Choose a TIFF by dragging it onto Run Super Resolution.bat.") from error


def normalize_band_name(name: str | None) -> str | None:
    if not name:
        return None
    normalized = name.strip().upper().replace("_", "").replace(" ", "")
    match = re.fullmatch(r"B0*(\d+)", normalized)
    return f"B{int(match.group(1))}" if match else normalized


def select_band_indexes(dataset: rasterio.DatasetReader) -> tuple[int, ...]:
    names = [normalize_band_name(name) for name in dataset.descriptions]
    named_indexes = {name: index + 1 for index, name in enumerate(names) if name}

    if all(band in named_indexes for band in REQUIRED_BANDS):
        return tuple(named_indexes[band] for band in REQUIRED_BANDS)

    if dataset.count == 4 and not any(names):
        # Four unnamed bands are accepted only in the standard RGBN order.
        return (1, 2, 3, 4)

    if dataset.count == 13 and not any(names):
        # Standard Sentinel-2 order: B1, B2, B3, B4, B5, B6, B7, B8, ... B12.
        return (4, 3, 2, 8)

    found = ", ".join(name or "unnamed" for name in names)
    raise ValueError(
        "This model needs real Sentinel-2 B4, B3, B2, and B8 bands. "
        f"This file has {dataset.count} band(s): {found or 'unnamed'}. "
        "Band-labelled TIFFs are accepted in any order, as are unnamed standard 13-band "
        "Sentinel-2 TIFFs. Export/download it with B8 (NIR) included; RGB-only TIFFs "
        "cannot be converted reliably."
    )


def prepare_rgbn(source: Path, workspace: Path, min_tile_size: int = 128) -> tuple[Path, int, int]:
    staged = workspace / "input_rgbn.tif"
    with rasterio.open(source) as dataset:
        indexes = select_band_indexes(dataset)
        source_width, source_height = dataset.width, dataset.height
        padded_width = max(source_width, min_tile_size)
        padded_height = max(source_height, min_tile_size)
        profile = dataset.profile.copy()
        profile.update(
            count=4,
            width=padded_width,
            height=padded_height,
            compress="deflate",
            predictor=2,
        )

        # OpenSR's Sentinel-2 model expects the common 0-10000 reflectance scale.
        selected_data = np.stack([dataset.read(index) for index in indexes]).astype(np.float32)
        minimum = min(float(data.min()) for data in selected_data)
        maximum = max(float(data.max()) for data in selected_data)
        scale = 10000.0 if 0.0 <= minimum and maximum <= 1.5 else 1.0
        selected_data *= scale
        if scale != 1.0 or padded_width != source_width or padded_height != source_height:
            profile.update(dtype="float32")

        if padded_width != source_width or padded_height != source_height:
            pad_mode = "reflect" if source_width > 1 and source_height > 1 else "edge"
            selected_data = np.pad(
                selected_data,
                ((0, 0), (0, padded_height - source_height), (0, padded_width - source_width)),
                mode=pad_mode,
            )
            print(
                f"Input is {source_width}x{source_height}; padded to "
                f"{padded_width}x{padded_height} for the {min_tile_size}px model tile."
            )

        with rasterio.open(staged, "w", **profile) as output:
            for destination_index, data in enumerate(selected_data, start=1):
                output.write(data, destination_index)
                output.set_band_description(destination_index, REQUIRED_BANDS[destination_index - 1])
        if scale != 1.0:
            print("Input uses 0-1 reflectance; converted it to the 0-10000 Sentinel-2 scale.")
    return staged, source_width, source_height


def crop_output_to_source_extent(path: Path, source_width: int, source_height: int, factor: int = 4) -> None:
    """Remove right/bottom model padding while preserving the source footprint."""
    target_width = source_width * factor
    target_height = source_height * factor
    temporary = path.with_name(f"{path.stem}_cropped{path.suffix}")

    with rasterio.open(path) as source:
        if target_width > source.width or target_height > source.height:
            raise ValueError(
                f"SR output {source.width}x{source.height} is smaller than the "
                f"expected source footprint {target_width}x{target_height}."
            )
        if target_width == source.width and target_height == source.height:
            return

        window = Window(0, 0, target_width, target_height)
        data = source.read(window=window)
        profile = source.profile.copy()
        profile.update(
            width=target_width,
            height=target_height,
            transform=source.window_transform(window),
        )
        descriptions = source.descriptions
        colorinterp = source.colorinterp
        tags = source.tags()

    with rasterio.open(temporary, "w", **profile) as destination:
        destination.write(data)
        for index, description in enumerate(descriptions, start=1):
            if description:
                destination.set_band_description(index, description)
        destination.colorinterp = colorinterp
        if tags:
            destination.update_tags(**tags)
    temporary.replace(path)


def label_output(path: Path) -> None:
    with rasterio.open(path, "r+") as dataset:
        for index, band_name in enumerate(REQUIRED_BANDS, start=1):
            dataset.set_band_description(index, band_name)


def load_ldsrs2_model(device: str = "cpu", model_version: str = "v1"):
    """Load the canonical v1 model or the explicitly selected v2 research preview."""
    if model_version == "v1":
        local_config_path = ROOT / "opensr_model" / "configs" / "config_10m.yaml"
        ckpt_path = ROOT / "opensr-ldsrs2_v1_0_0.ckpt"
    elif model_version == "v2-preview":
        local_config_path = ROOT / "v2.0.0" / "inference.yaml"
        ckpt_path = ROOT / "opensr-ldsrs2_v2_0_0_preview.ckpt"
    else:
        raise ValueError(f"Unsupported LDSR-S2 model version: {model_version}")

    if not local_config_path.is_file() or not ckpt_path.is_file():
        raise FileNotFoundError(
            f"The {model_version} model requires {local_config_path.name} and "
            f"{ckpt_path.name} in {ROOT}."
        )

    from omegaconf import OmegaConf
    import opensr_model

    config = OmegaConf.load(local_config_path)
    if model_version == "v1":
        if getattr(config.denoiser_settings, "sampling_steps", None) is None:
            config.denoiser_settings.sampling_steps = 100
        if getattr(config.denoiser_settings, "sampling_eta", None) is None:
            config.denoiser_settings.sampling_eta = 0.95

    model = opensr_model.SRLatentDiffusion(config, device=device)
    model.load_pretrained(str(ckpt_path))
    return model

def run(source: Path, model_version: str = "v1", window_size: int | None = None) -> Path:
    source = source.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Choose an existing .tif or .tiff file.")

    OUTPUTS.mkdir(exist_ok=True)
    destination = OUTPUTS / f"{source.stem}_{model_version}_sr.tif"
    workspace = Path(tempfile.mkdtemp(prefix="opensr_", dir=ROOT))
    # V2's global attention layer scales quadratically with tile dimensions.
    # A 64px LR tile keeps its peak VRAM low on 8GB GPUs; v1 retains its
    # established 128px tile unless explicitly overridden.
    tile_size = window_size or (64 if model_version == "v2-preview" else 128)
    keep_workspace = False

    try:
        prepared, source_width, source_height = prepare_rgbn(source, workspace, min_tile_size=tile_size)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sampler = "DPM-Solver++" if model_version == "v2-preview" else "DDIM"
        steps = 3 if model_version == "v2-preview" else 100
        print(
            f"Using LDSR-S2 {model_version} with {sampler} ({steps} steps) "
            f"on {device.upper()}. Preparing {source.name}..."
        )
        model = load_ldsrs2_model(device=device, model_version=model_version)
        # opensr-utils defaults to batches of 16 tiles, which can exhaust GPU
        # memory on attention-heavy v2 even when windows are tiled. Limit the
        # DataLoader to one patch at a time for this inference subprocess.
        import opensr_utils.pipeline as opensr_pipeline
        base_datamodule = opensr_pipeline.PredictionDataModule

        class SingleTilePredictionDataModule(base_datamodule):
            def __init__(self, *args, **kwargs):
                kwargs["batch_size"] = 1
                kwargs["num_workers"] = 0
                super().__init__(*args, **kwargs)

        opensr_pipeline.PredictionDataModule = SingleTilePredictionDataModule
        try:
            pipeline = opensr_utils.large_file_processing(
                root=str(prepared),
                model=model,
                window_size=(tile_size, tile_size),
                factor=4,
                overlap=12,
                eliminate_border_px=2,
                device=device,
                gpus=0 if device == "cuda" else None,
                save_preview=False,
                debug=False,
            )
        finally:
            opensr_pipeline.PredictionDataModule = base_datamodule
        # Prefer the path reported by opensr-utils, which can vary across
        # package versions, and keep the workspace path as a compatibility
        # fallback.
        output_candidates = [
            getattr(pipeline, "final_sr_path", None),
            Path(getattr(pipeline, "output_dir", "")) / "sr.tif"
            if getattr(pipeline, "output_dir", None) else None,
            workspace / "sr.tif",
        ]
        generated = next(
            (Path(candidate).expanduser().resolve() for candidate in output_candidates
             if candidate and Path(candidate).expanduser().is_file()),
            None,
        )
        if generated is None:
            keep_workspace = True
            temp_dir = Path(getattr(pipeline, "temp_folder", workspace / "temp"))
            index_path = temp_dir / "index.json"
            try:
                index_summary = (
                    f"{len(json.loads(index_path.read_text()).get('entries', []))} stitched patches"
                    if index_path.is_file() else "index.json missing"
                )
            except Exception as error:
                index_summary = f"could not read index.json: {error}"
            remaining_files = sorted(
                str(path.relative_to(workspace)) for path in workspace.rglob("*") if path.is_file()
            )[:30]
            log_paths = sorted(workspace.glob("logs*/log.txt"))
            log_tail = ""
            if log_paths:
                try:
                    log_tail = "\n".join(log_paths[-1].read_text(errors="replace").splitlines()[-20:])
                except OSError as error:
                    log_tail = f"Could not read OpenSR log: {error}"
            raise RuntimeError(
                "OpenSR inference returned without a discoverable output TIFF. "
                f"Pipeline final_sr_path={getattr(pipeline, 'final_sr_path', None)!r}; "
                f"output_dir={getattr(pipeline, 'output_dir', None)!r}; "
                f"debug={getattr(pipeline, 'debug', 'unknown')!r}; "
                f"patch index: {index_summary}; files in workspace: {remaining_files}; "
                f"OpenSR log tail: {log_tail or 'no log file found'}"
            )
        if destination.exists():
            destination.unlink()
        shutil.move(str(generated), str(destination))
        crop_output_to_source_extent(destination, source_width, source_height, factor=4)
        label_output(destination)
        return destination
    finally:
        if keep_workspace:
            print(f"OpenSR diagnostic workspace retained at: {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Super-resolve one Sentinel-2 RGBN GeoTIFF.")
    parser.add_argument("input", nargs="?", help="TIFF to process")
    parser.add_argument(
        "--model-version",
        choices=("v1", "v2-preview"),
        default="v1",
        help="Checkpoint version (v1 is canonical; v2-preview is experimental).",
    )
    parser.add_argument(
        "--window-size", type=int, choices=(64, 128), default=None,
        help="Low-resolution patch size. Defaults to 64 for v2-preview and 128 for v1.",
    )
    arguments = parser.parse_args()
    source = Path(arguments.input) if arguments.input else choose_input()
    if source is None:
        print("No file selected.")
        return 0

    try:
        result = run(source, model_version=arguments.model_version, window_size=arguments.window_size)
    except Exception as error:
        import traceback
        traceback.print_exc()
        print(f"\nCould not process the TIFF:\n{error}", file=sys.stderr)
        return 1

    print(f"\nDone. Your super-resolved TIFF is here:\n{result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
