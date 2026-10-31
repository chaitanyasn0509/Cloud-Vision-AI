"""
Satellite Cloud Removal & Reconstruction - Streamlit Application.

Integrates the canonical backend inference pipeline with a full user-facing
satellite acquisition workflow:
  Tab 1: Live Acquisition (Location Search -> Geocoding -> AOI Generation -> CDSE Catalogue -> Acquisition -> Harmonization -> ECRformer)
  Tab 2: Cached Phase 2 Baseline (Pre-validated CDSE Data / Custom Uploads -> ECRformer)
"""
from __future__ import annotations

import datetime
import io
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling as RioRes
from PIL import Image
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.processing import (
    InferenceService,
    InferenceConfig,
    InferenceResult,
    InferenceDiagnostics,
    HarmonizationConfig,
    HarmonizedPair,
    harmonize_pair,
    get_device,
    OpenSRService,
    OpenSRConfig,
    OpenSRResult,
    DSen2CRService,
    DSen2CRResult,
)
from backend.services.processing.metrics_ui import (
    render_ecrformer_metrics_ui,
    render_opensr_metrics_ui,
)
from backend.services.satellite.base import BoundingBox, S2Scene, S1Scene, best_s1_scene_index
from backend.services.satellite import (
    ClearReferenceService,
    ClearReferenceCandidate,
)
from backend.services.processing.reference_adapter import ValidatedReference
from backend.services.processing.reference_adapter import load_and_validate_reference
from backend.services.processing.evaluation_metrics import OPENSR_BAND_NAMES, compute_reference_based_metrics
from backend.services.satellite.geocoding import (
    GeocodedLocation,
    geocode_location,
    bbox_from_center,
    validate_aoi,
)
from backend.services.satellite.acquisition_cache import (
    S2_BAND_NAMES,
    S1_BAND_NAMES,
    is_s2_cached,
    is_s1_cached,
    get_s2_cache_dir,
    get_s1_cache_dir,
    write_cache_metadata,
)
from visualization import to_rgb, sar_preview
from ui.theme import inject_css, toggle_theme, get_theme
from ui.landing import render_landing_page
from ui.cnn_workspace import render_cnn_workspace
from backend.services.processing.persistence import record_run, get_recent_runs

CACHE_DIR = ROOT / "acquisition" / "data_cache"
HARM_OUTPUT_BASE = ROOT / "harmonization" / "output"
DEFAULT_S1 = HARM_OUTPUT_BASE / "s1_harmonized_10m.tif"
DEFAULT_S2 = HARM_OUTPUT_BASE / "s2_harmonized_10m.tif"
DEFAULT_MASK = HARM_OUTPUT_BASE / "validity_mask_10m.tif"
DEFAULT_CKPT = ROOT / "model.ckpt"

# ---- Page config ----
st.set_page_config(
    page_title="CloudVisionAI — Satellite Image Restoration",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Theme & Session State ----
if "cvai_theme" not in st.session_state:
    st.session_state["cvai_theme"] = "dark"
if "active_view" not in st.session_state:
    st.session_state["active_view"] = "landing"

# Preserve credentials in os.environ across all reruns
if "cdse_cid" in st.session_state and st.session_state["cdse_cid"]:
    os.environ["CDSE_CLIENT_ID"] = st.session_state["cdse_cid"]
if "cdse_csec" in st.session_state and st.session_state["cdse_csec"]:
    os.environ["CDSE_CLIENT_SECRET"] = st.session_state["cdse_csec"]

# Inject full CloudVisionAI theme stylesheet
inject_css()

# Render landing page when active view is "landing"
if st.session_state.get("active_view", "landing") == "landing":
    render_landing_page()
    st.stop()

# ===========================================================================
# WORKSPACE VIEW
# ===========================================================================

# ---- Sidebar Configuration ----
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
            <span style="font-size:1.5rem;">🛰️</span>
            <div>
                <span class="cvai-brand" style="font-size:1.2rem;">CloudVisionAI</span><br>
                <span class="cvai-brand-sub">Configuration</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cid = os.environ.get("CDSE_CLIENT_ID", "").strip()
    csec = os.environ.get("CDSE_CLIENT_SECRET", "").strip()

    st.markdown(
        """
        <div style="font-family:'Space Grotesk', sans-serif; font-size:1.1rem; font-weight:700; margin-top:8px; margin-bottom:4px;">
            CDSE Authentication
        </div>
        """,
        unsafe_allow_html=True,
    )

    if cid and csec:
        masked_cid = f"{cid[:4]}...{cid[-4:]}" if len(cid) > 8 else "***"
        st.success(f"**Status:** Configured  \n`ID: {masked_cid}`")
        with st.expander("⚙️ Update CDSE Credentials", expanded=False):
            u_cid = st.text_input(
                "CDSE Client ID",
                value=st.session_state.get("cdse_cid", cid),
                placeholder="e.g. 5fa7b2c1-...",
                key="input_cdse_cid",
            )
            u_csec = st.text_input(
                "CDSE Client Secret",
                value=st.session_state.get("cdse_csec", csec),
                type="password",
                placeholder="••••••••••••••••",
                key="input_cdse_csec",
            )
            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                if st.button("Apply", key="btn_apply_creds", type="primary", use_container_width=True):
                    if u_cid.strip() and u_csec.strip():
                        os.environ["CDSE_CLIENT_ID"] = u_cid.strip()
                        os.environ["CDSE_CLIENT_SECRET"] = u_csec.strip()
                        st.session_state["cdse_cid"] = u_cid.strip()
                        st.session_state["cdse_csec"] = u_csec.strip()
                        st.session_state.pop("catalogue_provider", None)
                        st.session_state.pop("cnn_catalogue_provider", None)
                        st.session_state.pop("comparison_catalogue_provider", None)
                        st.success("Credentials saved to active session.")
                        st.rerun()
                    else:
                        st.error("Both fields required.")
            with c_btn2:
                if st.button("Verify", key="btn_test_creds", use_container_width=True):
                    from backend.auth.cdse_auth import test_credentials
                    test_cid = u_cid.strip() or cid
                    test_csec = u_csec.strip() or csec
                    if test_cid and test_csec:
                        with st.spinner("Connecting to CDSE OAuth2 endpoint..."):
                            ok, msg = test_credentials(test_cid, test_csec)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
                    else:
                        st.error("Enter Client ID and Secret to test.")

            if st.button("Clear Credentials", key="btn_clear_creds", use_container_width=True):
                os.environ.pop("CDSE_CLIENT_ID", None)
                os.environ.pop("CDSE_CLIENT_SECRET", None)
                st.session_state.pop("cdse_cid", None)
                st.session_state.pop("cdse_csec", None)
                st.session_state.pop("catalogue_provider", None)
                st.session_state.pop("cnn_catalogue_provider", None)
                st.session_state.pop("comparison_catalogue_provider", None)
                st.info("Credentials cleared.")
                st.rerun()
    else:
        st.warning("**Status:** Not Configured")
        st.caption("Enter your CDSE OAuth2 credentials for live Sentinel-1 & 2 search and acquisition:")
        u_cid = st.text_input(
            "CDSE Client ID",
            value=st.session_state.get("cdse_cid", ""),
            placeholder="e.g. 5fa7b2c1-...",
            key="input_cdse_cid",
        )
        u_csec = st.text_input(
            "CDSE Client Secret",
            value=st.session_state.get("cdse_csec", ""),
            type="password",
            placeholder="••••••••••••••••",
            key="input_cdse_csec",
        )
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("Apply", key="btn_apply_creds", type="primary", use_container_width=True):
                if u_cid.strip() and u_csec.strip():
                    os.environ["CDSE_CLIENT_ID"] = u_cid.strip()
                    os.environ["CDSE_CLIENT_SECRET"] = u_csec.strip()
                    st.session_state["cdse_cid"] = u_cid.strip()
                    st.session_state["cdse_csec"] = u_csec.strip()
                    st.session_state.pop("catalogue_provider", None)
                    st.session_state.pop("cnn_catalogue_provider", None)
                    st.session_state.pop("comparison_catalogue_provider", None)
                    st.success("Credentials saved to active session.")
                    st.rerun()
                else:
                    st.error("Both fields required.")
        with c_btn2:
            if st.button("Verify", key="btn_test_creds", use_container_width=True):
                from backend.auth.cdse_auth import test_credentials
                if u_cid.strip() and u_csec.strip():
                    with st.spinner("Connecting to CDSE OAuth2 endpoint..."):
                        ok, msg = test_credentials(u_cid.strip(), u_csec.strip())
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.error("Enter Client ID and Secret to test.")

    device_obj = get_device()
    device_str = "CUDA (RTX 3050)" if "cuda" in str(device_obj) else "CPU"
    st.info(f"**Compute Device:** {device_str}")

    if DEFAULT_CKPT.exists():
        st.success(f"**Checkpoint:** model.ckpt ({DEFAULT_CKPT.stat().st_size / 1e6:.1f} MB)")
    else:
        st.error(f"**Checkpoint missing:** {DEFAULT_CKPT}")

    st.divider()
    st.subheader("Inference Settings")
    tile_size = st.select_slider("Tile Size", options=[64, 128, 256], value=128)
    overlap = st.select_slider("Tile Overlap", options=[16, 32, 64], value=32)
    brightness = st.slider("RGB Brightness", 1.0, 5.0, 2.5, 0.1)

    # SQLite Run History
    st.divider()
    with st.expander("📜 Recent Run History (SQLite)", expanded=False):
        recent_runs = get_recent_runs(limit=5)
        if recent_runs:
            for r in recent_runs:
                st.markdown(
                    f"**`{r['run_id']}`** ({r['mode']})  \n"
                    f"<span style='font-size:0.75rem; color:#8B949E;'>"
                    f"{r['created_at'][:19].replace('T', ' ')} · {r.get('dimensions', 'N/A')} · {r.get('execution_time_s', 0):.2f}s"
                    f"</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No persistent run records yet.")

# ---- Singleton Inference Service ----
@st.cache_resource
def get_service():
    return InferenceService(checkpoint_path=DEFAULT_CKPT)

service = get_service()


@st.cache_resource
def get_dsen2cr_service():
    return DSen2CRService(project_root=ROOT)


def get_model_input_pair():
    """Reuse the latest CDSE harmonized pair, falling back to the Rome benchmark."""
    pair = st.session_state.get("live_harm_pair")
    if pair is not None and Path(pair.s1_path).is_file() and Path(pair.s2_path).is_file():
        mask_path = pair.validity_mask_path
        return Path(pair.s1_path), Path(pair.s2_path), Path(mask_path) if mask_path else None, "Latest CDSE acquisition"
    saved = get_available_model_input_pairs()
    if saved:
        return saved[0][1], saved[0][2], saved[0][3], saved[0][0]
    if DEFAULT_S1.is_file() and DEFAULT_S2.is_file():
        return DEFAULT_S1, DEFAULT_S2, DEFAULT_MASK if DEFAULT_MASK.is_file() else None, "Cached Rome harmonized data"
    return None, None, None, "No harmonized inputs available"


def get_available_model_input_pairs():
    """List existing harmonized CDSE/session pairs so model tabs can reuse them."""
    pairs = []
    seen = set()

    def add(label, s1_path, s2_path, mask_path=None):
        s1_path, s2_path = Path(s1_path), Path(s2_path)
        mask_path = Path(mask_path) if mask_path else None
        if not s1_path.is_file() or not s2_path.is_file():
            return
        identity = (str(s1_path.resolve()), str(s2_path.resolve()))
        if identity in seen:
            return
        seen.add(identity)
        pairs.append((label, s1_path, s2_path, mask_path if mask_path and mask_path.is_file() else None))

    pair = st.session_state.get("live_harm_pair")
    if pair is not None:
        add("Current CDSE acquisition", pair.s1_path, pair.s2_path, pair.validity_mask_path)

    sessions_dir = HARM_OUTPUT_BASE / "sessions"
    if sessions_dir.is_dir():
        session_dirs = sorted(
            (directory for directory in sessions_dir.iterdir() if directory.is_dir()),
            key=lambda directory: directory.stat().st_mtime,
            reverse=True,
        )
        for directory in session_dirs:
            s1_path = directory / "s1_harmonized_10m.tif"
            s2_path = directory / "s2_harmonized_10m.tif"
            mask_path = directory / "validity_mask_10m.tif"
            if s1_path.is_file() and s2_path.is_file():
                add("Saved harmonized CDSE session · {}".format(directory.name), s1_path, s2_path, mask_path)
    if DEFAULT_S1.is_file() and DEFAULT_S2.is_file():
        add("Cached Rome benchmark", DEFAULT_S1, DEFAULT_S2, DEFAULT_MASK)
    return pairs


def render_model_input_pair_selector(widget_key):
    pairs = get_available_model_input_pairs()
    if not pairs:
        return None, None, None, "No harmonized inputs available"
    labels = [pair[0] for pair in pairs]
    selected = st.selectbox("Harmonized Sentinel-1/2 input", labels, key=widget_key) if len(labels) > 1 else labels[0]
    match = next(pair for pair in pairs if pair[0] == selected)
    return match[1], match[2], match[3], match[0]


def get_selected_cdse_pair():
    """Return the current AOI and selected S1/S2 catalogue records, if any."""
    s2_scenes = st.session_state.get("s2_scenes", [])
    s1_scenes = st.session_state.get("s1_scenes", [])
    s2_index = st.session_state.get("selected_s2_idx")
    s1_index = st.session_state.get("selected_s1_idx")
    bbox = st.session_state.get("selected_bbox")
    if bbox is None or s2_index is None or s1_index is None:
        return None
    if s2_index >= len(s2_scenes) or s1_index >= len(s1_scenes):
        return None
    return bbox, s2_scenes[s2_index], s1_scenes[s1_index]


def get_selected_cnn_cdse_pair():
    s2_scenes = st.session_state.get("cnn_s2_scenes", [])
    s1_scenes = st.session_state.get("cnn_s1_scenes", [])
    s2_index = st.session_state.get("cnn_selected_s2_idx")
    s1_index = st.session_state.get("cnn_selected_s1_idx")
    bbox = st.session_state.get("cnn_selected_bbox")
    if bbox is None or s2_index is None or s1_index is None:
        return None
    if s2_index >= len(s2_scenes) or s1_index >= len(s1_scenes):
        return None
    return bbox, s2_scenes[s2_index], s1_scenes[s1_index]


def acquire_and_harmonize_selected_pair(bbox, s2_scene, s1_scene, output_px, session_prefix, provider_state_key):
    """Download one CDSE S1/S2 pair and run the shared Phase 2 harmonizer."""
    output_size = (output_px, output_px)
    session_id = session_prefix + "_" + str(uuid.uuid4())[:8]
    output_dir = HARM_OUTPUT_BASE / "sessions" / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    provider = st.session_state.get(provider_state_key)
    if provider is None:
        from backend.services.satellite.cdse_sentinelhub import CDSESentinelHubProvider
        provider = CDSESentinelHubProvider()
        st.session_state[provider_state_key] = provider

    s2_cache_dir = get_s2_cache_dir(CACHE_DIR, bbox, s2_scene.utc_date(), output_size)
    s2_cache_dir.mkdir(parents=True, exist_ok=True)
    s2_path = s2_cache_dir / "s2.tif"
    if not is_s2_cached(CACHE_DIR, bbox, s2_scene.utc_date(), output_size):
        provider.download_sentinel2(
            bbox=bbox, target_date=s2_scene.utc_date(), output_path=str(s2_path), output_size_px=output_size,
        )
        write_cache_metadata(
            s2_cache_dir, "sentinel-2-l1c", bbox, s2_scene.utc_date(), S2_BAND_NAMES, output_size,
            extra={"scene_id": s2_scene.scene_id, "cloud_cover": s2_scene.cloud_cover},
        )

    s1_variant = "sigma0_ellipsoid_v1"
    s1_cache_dir = get_s1_cache_dir(
        CACHE_DIR, bbox, s1_scene.utc_date(), output_size,
        processing_variant=s1_variant,
    )
    s1_cache_dir.mkdir(parents=True, exist_ok=True)
    s1_path = s1_cache_dir / "s1.tif"
    if not is_s1_cached(
        CACHE_DIR, bbox, s1_scene.utc_date(), output_size,
        processing_variant=s1_variant,
    ):
        provider.download_sentinel1(
            bbox=bbox, target_date=s1_scene.utc_date(), output_path=str(s1_path),
            output_size_px=output_size, back_coeff="SIGMA0_ELLIPSOID",
        )
        write_cache_metadata(
            s1_cache_dir, "sentinel-1-grd", bbox, s1_scene.utc_date(), S1_BAND_NAMES, output_size,
            extra={
                "scene_id": s1_scene.scene_id,
                "polarizations": s1_scene.polarizations,
                "orbit_direction": s1_scene.orbit_direction,
                "instrument_mode": s1_scene.instrument_mode,
                "backscatter_coefficient": "SIGMA0_ELLIPSOID",
            },
            processing_variant=s1_variant,
        )

    harm_pair = harmonize_pair(
        s2_input_path=s2_path,
        s1_input_path=s1_path,
        output_dir=output_dir,
        config=HarmonizationConfig(
            pixel_size=10.0,
            s2_resampling=RioRes.bilinear,
            s1_resampling=RioRes.bilinear,
            save_previews=True,
            save_validity_mask=True,
        ),
    )
    return harm_pair


def acquire_and_harmonize_for_cnn():
    """Acquire the CNN tab's selected pair without changing Former Model state."""
    selected = get_selected_cnn_cdse_pair()
    if selected is None:
        raise ValueError("Select one Sentinel-2 and one Sentinel-1 scene in CNN Model → Live Acquisition first.")
    bbox, s2_scene, s1_scene = selected
    output_px = int(st.session_state.get("cnn_slider_raw_grid_px", 256))
    harm_pair = acquire_and_harmonize_selected_pair(
        bbox, s2_scene, s1_scene, output_px, "cnn", "cnn_catalogue_provider",
    )
    st.session_state["cnn_harm_pair"] = harm_pair
    st.session_state["cnn_live_s2_meta"] = s2_scene
    st.session_state["cnn_live_bbox"] = bbox
    return harm_pair


def normalize_sen12ms_sar_upload(sar_bytes: bytes, optical_bytes: bytes):
    """Recover a common SEN12MS-CR TIFF layout where VV/VH are packed oddly.

    Some distributed 256x256x2 SAR patches are encoded with TIFF axes
    (256, 2, 256), making rasterio report 256 bands and a 2-pixel-wide image.
    Their values are already in dB. Other SAR GeoTIFF layouts pass through.
    """
    with rasterio.io.MemoryFile(sar_bytes) as sar_mem, rasterio.io.MemoryFile(optical_bytes) as optical_mem:
        with sar_mem.open() as sar_src, optical_mem.open() as optical_src:
            is_sen12ms_layout = (
                sar_src.count == 256 and sar_src.height == 256 and sar_src.width == 2
                and sar_src.crs is None and (optical_src.width, optical_src.height) == (256, 256)
            )
            if not is_sen12ms_layout:
                return sar_bytes, False
            optical_profile = optical_src.profile.copy()

    try:
        import tifffile
    except ImportError as exc:
        raise ValueError("This SEN12MS-CR SAR TIFF layout requires tifffile support in the app environment.") from exc
    raw = tifffile.imread(io.BytesIO(sar_bytes))
    if raw.shape != (256, 2, 256):
        raise ValueError(f"Unexpected SEN12MS-CR SAR array shape: {raw.shape}.")
    # Recover each scanline as 256 interleaved VV/VH pixels.
    sar = np.ascontiguousarray(raw.reshape(256, 256, 2).transpose(2, 0, 1), dtype=np.float32)
    finite = sar[np.isfinite(sar)]
    if finite.size == 0 or float(finite.min()) < -60.0 or float(finite.max()) > 10.0:
        raise ValueError("SEN12MS-CR SAR values are outside the expected VV/VH dB range (-60 to 10 dB).")
    profile = optical_profile.copy()
    profile.update(driver="GTiff", count=2, width=256, height=256, dtype="float32", nodata=None, compress="deflate")
    with rasterio.io.MemoryFile() as normalized_mem:
        with normalized_mem.open(**profile) as dst:
            dst.write(sar)
            dst.set_band_description(1, "VV")
            dst.set_band_description(2, "VH")
        return normalized_mem.read(), True


def evaluate_opensr_against_ground_truth(sr_result, ground_truth_input, target_profile):
    """Score 4-band OpenSR output against matching S2 ground truth on its 10 m grid."""
    reference = load_and_validate_reference(
        reference_input=ground_truth_input,
        target_profile=target_profile,
        scale_mode="auto",
        reference_type="paired_ground_truth",
    )
    with rasterio.open(sr_result.sr_geotiff_path) as source:
        if source.count != 4:
            raise ValueError(f"OpenSR output must contain four bands; found {source.count}.")
        high_resolution = source.read().astype(np.float32)

    _, height, width = reference.array.shape
    if high_resolution.shape[1:] != (height * 4, width * 4):
        raise ValueError(
            "OpenSR output is not exactly 4× the aligned ground-truth grid; "
            "cannot calculate the 10 m ground-truth comparison."
        )
    downsampled = high_resolution.reshape(4, height, 4, width, 4).mean(axis=(2, 4))
    if float(np.nanmax(downsampled)) > 1.5:
        downsampled = np.clip(downsampled / 10000.0, 0.0, 1.0)
    else:
        downsampled = np.clip(downsampled, 0.0, 1.0)

    # OpenSR order is B04, B03, B02, B08; canonical S2 order is B01..B12.
    reference_bands = reference.array[[3, 2, 1, 7]]
    return compute_reference_based_metrics(
        prediction=downsampled,
        target=reference_bands,
        valid_mask=reference.valid_mask,
        band_names=OPENSR_BAND_NAMES,
        data_range=1.0,
        reference_type="paired_ground_truth",
        reference_source=reference.source_name,
        rgb_band_indices=(0, 1, 2),
    )


def render_comparison_metric_charts(metrics_by_model, chart_key_prefix: str) -> None:
    """Show each metric on its own scale so unlike units remain readable."""
    chart_specs = (
        ("MAE", "aggregate_mae", "Lower is better"),
        ("RMSE", "aggregate_rmse", "Lower is better"),
        ("PSNR (dB)", "aggregate_psnr_db", "Higher is better"),
        ("SSIM (RGB)", "ssim_rgb", "Higher is better"),
        ("SAM (°)", "sam_degrees", "Lower is better"),
    )
    available = {
        name: metric
        for name, metric in metrics_by_model.items()
        if metric is not None and not isinstance(metric, Exception)
    }
    if not available:
        return

    st.markdown("#### Metric comparison charts")
    for row_start in range(0, len(chart_specs), 3):
        row_specs = chart_specs[row_start : row_start + 3]
        chart_columns = st.columns(len(row_specs))
        for column, (label, attribute, direction) in zip(chart_columns, row_specs):
            values = []
            for model_name in ("Former Model", "CNN"):
                metric = available.get(model_name)
                value = getattr(metric, attribute, None) if metric is not None else None
                if value is not None and np.isfinite(value):
                    values.append({"Model": model_name, "Score": float(value)})
            with column:
                st.caption(f"{label} · {direction}")
                if values:
                    chart_data = pd.DataFrame(values).set_index("Model")
                    st.bar_chart(
                        chart_data,
                        height=190,
                        y_label=label,
                        alt=f"{chart_key_prefix}: {label} scores for Former Model and CNN",
                    )
                else:
                    st.caption("Not available for these results.")


def render_cnn_result(
    result: DSen2CRResult,
    image_id: str,
    key_prefix: str,
    sr_state_key: str,
    harm_pair: Optional[HarmonizedPair] = None,
    bbox: Optional[BoundingBox] = None,
    target_date: Optional[str] = None,
) -> None:
    """Render the same reconstruction, export, evaluation and OpenSR sections as Former Model."""
    diagnostics = result.diagnostics
    st.divider()
    st.subheader("Reconstruction Results")
    st.success(f"DSen2-CR Reconstruction Successful ({diagnostics.execution_time_seconds:.2f}s on {diagnostics.device})")

    model_prediction = getattr(result, "model_prediction", result.prediction)
    cloud_score = getattr(result, "cloud_score", None)
    original_reflectance = getattr(result, "cloudy_reflectance", None)
    output_controls = st.columns([1, 2])
    with output_controls[0]:
        output_mode = st.radio(
            "CNN output",
            ["Cloud-guided blend", "Raw DSen2-CR"],
            horizontal=True,
            key=f"{key_prefix}_output_mode",
            help="The cloud-guided blend preserves original pixels outside likely cloud areas. Raw mode shows the network output everywhere.",
        )
    threshold = 0.4
    if cloud_score is not None and original_reflectance is not None and output_mode == "Cloud-guided blend":
        with output_controls[1]:
            threshold = st.slider(
                "Cloud-score threshold (higher preserves more original detail)",
                min_value=0.20, max_value=0.70, value=0.40, step=0.05,
                key=f"{key_prefix}_cloud_score_threshold",
            )
        output_config = (output_mode, threshold)
        repaired, repair_weight = result.cloud_guided_prediction(threshold)
    else:
        output_config = (output_mode, None)
        repaired = model_prediction.copy()
        repair_weight = np.ones(model_prediction.shape[1:], dtype=np.float32)

    previous_config = st.session_state.get(f"{key_prefix}_output_config")
    if previous_config is not None and previous_config != output_config:
        st.session_state.pop(sr_state_key, None)
        st.session_state.pop(f"{key_prefix}_metrics_ref_metrics", None)
        result.reference_metrics = None
        diagnostics.reference_metrics = None
    st.session_state[f"{key_prefix}_output_config"] = output_config
    result.output_mode = output_mode
    result.cloud_score_threshold = threshold if output_mode == "Cloud-guided blend" else None
    result.prediction = repaired
    result.prediction_rgb = to_rgb(repaired, brightness=brightness, valid_mask=result.valid_mask)
    if output_mode == "Cloud-guided blend" and cloud_score is not None:
        st.caption(
            f"Cloud-guided blend applies CNN repair at full strength on high-score pixels and feathers the boundary; "
            f"average repair weight: {100 * float(repair_weight[result.valid_mask].mean()):.1f}% of valid pixels. "
            "The score is a spectral heuristic, not a validated cloud probability."
        )
    diagnostic_values = repaired[:, result.valid_mask]
    if diagnostic_values.size:
        diagnostics.min_val = float(diagnostic_values.min())
        diagnostics.max_val = float(diagnostic_values.max())
        diagnostics.mean_val = float(diagnostic_values.mean())
        diagnostics.per_band_stats = {
            name: {
                "min": float(repaired[index][result.valid_mask].min()),
                "max": float(repaired[index][result.valid_mask].max()),
                "mean": float(repaired[index][result.valid_mask].mean()),
            }
            for index, name in enumerate(S2_BAND_NAMES)
        }

    cards = st.columns(5)
    cards[0].metric("Compute Device", diagnostics.device.split("(")[0].strip().upper())
    cards[1].metric("Inference Time", f"{diagnostics.execution_time_seconds:.2f}s")
    cards[2].metric("Tiles Processed", diagnostics.tile_count)
    cards[3].metric("Valid Pixels", f"{diagnostics.valid_pixels:,}")
    cards[4].metric("Mean Reflectance", f"{diagnostics.mean_val:.4f}")

    input_quality = getattr(result, "input_quality", {})
    if input_quality:
        for warning in input_quality.get("warnings", []):
            st.warning(f"CNN input compatibility: {warning}")
        with st.expander("CNN Input Compatibility Diagnostics", expanded=bool(input_quality.get("warnings"))):
            st.caption(
                "DSen2-CR was trained with Sentinel-1 VV clipped to -25–0 dB, VH clipped to -32.5–0 dB, "
                "and Sentinel-2 DN clipped to 0–10,000. Values outside those limits are clipped before inference."
            )
            sar_rows = []
            for band, stats in input_quality.get("sar_bands", {}).items():
                sar_rows.append({
                    "Sensor band": band,
                    "p02 (dB)": f"{stats['p02_db']:.2f}",
                    "Median (dB)": f"{stats['p50_db']:.2f}",
                    "p98 (dB)": f"{stats['p98_db']:.2f}",
                    "Below training range": f"{100 * stats['below_training_range_fraction']:.1f}%",
                    "Above training range": f"{100 * stats['above_training_range_fraction']:.1f}%",
                })
            if sar_rows:
                st.dataframe(pd.DataFrame(sar_rows), use_container_width=True, hide_index=True)
            st.caption("These checks describe input compatibility; they are not reconstruction accuracy scores.")

    previews = st.columns(4)
    with previews[0]:
        st.image(result.cloudy_rgb, caption="Input Sentinel-2 Cloudy (RGB)", use_container_width=True)
    with previews[1]:
        st.image(result.sar_preview_img, caption="Harmonized Sentinel-1 SAR (VV/VH)", use_container_width=True)
    with previews[2]:
        if cloud_score is not None and output_mode == "Cloud-guided blend":
            score_preview = Image.fromarray(np.rint(np.clip(repair_weight, 0, 1) * 255).astype(np.uint8), mode="L")
            st.image(score_preview, caption="Cloud-guided CNN repair weight", use_container_width=True)
        else:
            st.info("Raw model output selected; CNN is applied across the whole valid image.")
    with previews[3]:
        output_caption = "Cloud-guided DSen2-CR reconstruction (RGB)" if output_mode == "Cloud-guided blend" else "Raw DSen2-CR output (RGB)"
        st.image(result.prediction_rgb, caption=output_caption, use_container_width=True)

    if harm_pair and harm_pair.preview_coreg_path and harm_pair.preview_coreg_path.exists():
        with st.expander("SAR / Optical Co-Registration Diagnostics", expanded=False):
            st.image(str(harm_pair.preview_coreg_path), caption="SAR edge overlay on Sentinel-2 optical image", use_container_width=True)

    st.subheader("Export & Download")
    date_token = target_date or image_id
    exports = st.columns(4)
    with exports[0]:
        st.download_button("Download 13-Band GeoTIFF", data=result.to_geotiff_bytes(), file_name=f"dsen2cr_reconstructed_{date_token}_13band.tif", mime="image/tiff", use_container_width=True, key=f"{key_prefix}_dl_tif")
    with exports[1]:
        st.download_button("Download RGB GeoTIFF (10 m)", data=result.to_rgb_geotiff_bytes(), file_name=f"dsen2cr_reconstructed_{date_token}_rgb_10m.tif", mime="image/tiff", use_container_width=True, key=f"{key_prefix}_dl_rgb")
    with exports[2]:
        st.download_button("Download Preview PNG", data=result.to_png_bytes(), file_name=f"dsen2cr_preview_{date_token}.png", mime="image/png", use_container_width=True, key=f"{key_prefix}_dl_png")
    with exports[3]:
        st.download_button("Download Diagnostics JSON", data=json.dumps(diagnostics.to_dict(), indent=2), file_name=f"dsen2cr_diagnostics_{date_token}.json", mime="application/json", use_container_width=True, key=f"{key_prefix}_dl_json")

    if bbox is not None and target_date:
        with st.expander("Find a Clear Reference for This AOI (CDSE Sentinel-2)", expanded=False):
            st.caption("Search for clear temporal Sentinel-2 scenes covering the same area, acquire one, and compare it against this reconstruction.")
            ref_cols = st.columns(3)
            with ref_cols[0]:
                window_days = st.slider("Search Window (±Days)", 5, 90, 30, 5, key=f"{key_prefix}_ref_window")
            with ref_cols[1]:
                max_cloud = st.slider("Max Scene Cloud %", 0.0, 80.0, 15.0, 5.0, key=f"{key_prefix}_ref_scene_cloud")
            with ref_cols[2]:
                max_aoi_cloud = st.slider("Max AOI Cloud %", 0.0, 50.0, 10.0, 2.0, key=f"{key_prefix}_ref_aoi_cloud")
            ref_collection = st.selectbox("Sentinel-2 Collection", ["sentinel-2-l2a", "sentinel-2-l1c"], key=f"{key_prefix}_ref_collection")
            if st.button("Search CDSE for Clear Reference Scenes", key=f"{key_prefix}_ref_search", use_container_width=True):
                try:
                    provider = st.session_state.get("cnn_catalogue_provider")
                    if provider is None:
                        from backend.services.satellite.cdse_sentinelhub import CDSESentinelHubProvider
                        provider = CDSESentinelHubProvider()
                        st.session_state["cnn_catalogue_provider"] = provider
                    candidates = ClearReferenceService(provider=provider).search_candidates(
                        bbox=bbox, target_date=target_date, window_days=int(window_days),
                        max_scene_cloud_cover=float(max_cloud), max_aoi_cloud_fraction=float(max_aoi_cloud),
                        max_candidates=6, collection=ref_collection,
                    )
                    st.session_state[f"{key_prefix}_ref_candidates"] = candidates
                    if not candidates:
                        st.info("No clear reference scenes matched these settings. Adjust the date window or cloud thresholds.")
                except Exception as exc:
                    st.error(f"CDSE clear-reference search failed: {exc}")
            candidates = st.session_state.get(f"{key_prefix}_ref_candidates", [])
            if candidates:
                labels = [candidate.display_label() for candidate in candidates]
                selected_index = st.selectbox("Candidate Reference Scene", range(len(candidates)), format_func=lambda index: labels[index], key=f"{key_prefix}_ref_candidate")
                candidate = candidates[selected_index]
                if st.button("Download, Align & Inspect Candidate Reference", key=f"{key_prefix}_ref_download", type="primary", use_container_width=True):
                    try:
                        reference = ClearReferenceService(provider=st.session_state.get("cnn_catalogue_provider")).acquire_and_align_reference(
                            candidate=candidate, bbox=bbox, target_profile=result.profile, scale_mode="auto",
                            output_size_px=(result.profile["width"], result.profile["height"]),
                        )
                        st.session_state[f"{key_prefix}_ref_result"] = reference
                        st.session_state[f"{key_prefix}_ref_candidate_meta"] = candidate
                    except Exception as exc:
                        st.error(f"Clear-reference acquisition failed: {exc}")
            reference = st.session_state.get(f"{key_prefix}_ref_result")
            reference_candidate = st.session_state.get(f"{key_prefix}_ref_candidate_meta")
            if reference is not None and reference_candidate is not None:
                ref_cols = st.columns(2)
                with ref_cols[0]:
                    st.image(ClearReferenceService.to_rgb_preview(reference), caption=f"Clear reference · {reference.acquisition_date}", use_container_width=True)
                with ref_cols[1]:
                    st.image(result.prediction_rgb, caption=f"DSen2-CR reconstruction · {target_date}", use_container_width=True)
                st.warning("This is a temporal reference from a different date; surface and atmospheric changes can affect the comparison.")
                if st.button("Calculate Temporal Reference Metrics", key=f"{key_prefix}_ref_metrics", use_container_width=True):
                    try:
                        result.evaluate_with_reference(reference)
                    except Exception as exc:
                        st.error(f"Reference metric calculation failed: {exc}")

    render_ecrformer_metrics_ui(
        result=result,
        image_id=image_id,
        key_prefix=f"{key_prefix}_metrics",
        allow_reference_upload=True,
        model_name="DSen2-CR",
    )

    st.divider()
    st.subheader("Optional Post-Processing: Diffusion Super-Resolution (OpenSR 4×)")
    st.caption("Apply the same LDSR-S2 diffusion pipeline as Former Model to B04, B03, B02 and B08 (10 m → 2.5 m).")
    sr_version = st.selectbox(
        "Diffusion checkpoint", ("v1", "v2-preview"),
        format_func=lambda version: "LDSR-S2 v1 (recommended)" if version == "v1" else "LDSR-S2 v2 research preview (experimental)",
        key=f"{key_prefix}_sr_version",
    )
    if sr_version == "v2-preview":
        st.caption("V2 preview uses DPM-Solver++ (3 steps) and back-projection. It remains experimental.")
    if st.button("✨ Generate Super-Resolved RGBN (OpenSR 4×)", key=f"{key_prefix}_run_sr", use_container_width=True):
        try:
            output_dir = ROOT / "reconstructed_images" / "sessions" / "dsen2cr" / image_id
            output_dir.mkdir(parents=True, exist_ok=True)
            source_tif = output_dir / "dsen2cr_cloud_free_13band.tif"
            result.save_geotiff(source_tif)
            with st.spinner(f"Executing OpenSR LDSR-S2 {sr_version} on the available device..."):
                sr_result = OpenSRService(OpenSRConfig(model_version=sr_version)).run(input_tif_path=source_tif, output_dir=output_dir)
            st.session_state[sr_state_key] = sr_result
        except Exception as exc:
            st.error(f"OpenSR Super-Resolution failed: {exc}")

    sr_result: Optional[OpenSRResult] = st.session_state.get(sr_state_key)
    if sr_result is not None:
        sr_cols = st.columns(2)
        with sr_cols[0]:
            st.image(result.prediction_rgb, caption="DSen2-CR 10 m Baseline (RGB)", use_container_width=True)
        with sr_cols[1]:
            st.image(sr_result.sr_rgb_preview, caption="OpenSR 2.5 m Super-Resolved RGB", use_container_width=True)
        sr_downloads = st.columns(2)
        with sr_downloads[0]:
            st.download_button("Download 4-Band SR GeoTIFF (2.5 m)", data=sr_result.to_geotiff_bytes(), file_name=f"dsen2cr_sr_{image_id}_4band_2_5m.tif", mime="image/tiff", use_container_width=True, key=f"{key_prefix}_sr_dl_tif")
        with sr_downloads[1]:
            st.download_button("Download 2.5 m RGB Preview PNG", data=sr_result.to_png_bytes(), file_name=f"dsen2cr_sr_{image_id}_preview.png", mime="image/png", use_container_width=True, key=f"{key_prefix}_sr_dl_png")
        render_opensr_metrics_ui(sr_res=sr_result, image_id=image_id, key_prefix=f"{key_prefix}_sr_metrics")


def run_cnn_inference(s1_path, s2_path, mask_path, result_key, sr_key, source_label, brightness,
                      bbox=None, target_date=None, harm_pair=None):
    service_cnn = get_dsen2cr_service()
    progress = st.progress(0, text="Preparing DSen2-CR GPU inference...")
    result = service_cnn.run(
        s1_input=s1_path,
        s2_cloudy_input=s2_path,
        validity_mask_input=mask_path,
        brightness=brightness,
        progress_callback=lambda done, total: progress.progress(
            int(100 * done / max(total, 1)), text=f"DSen2-CR GPU tiles: {done}/{total}"
        ),
    )
    st.session_state[result_key] = result
    st.session_state[f"{result_key}_source"] = source_label
    st.session_state[f"{result_key}_date"] = target_date
    st.session_state[f"{result_key}_bbox"] = bbox
    st.session_state[f"{result_key}_harm_pair"] = harm_pair
    st.session_state.pop(sr_key, None)
    for suffix in ("_ref_candidates", "_ref_result", "_ref_candidate_meta", "_metrics_ref_metrics"):
        st.session_state.pop(f"{result_key}{suffix}", None)
    return result

# Workspace Top Navigation Header
ws_col_brand, ws_col_actions = st.columns([6, 4])
with ws_col_brand:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px; margin-bottom: 2px;">
            <span style="font-size: 1.8rem;">🛰️</span>
            <div>
                <span class="cvai-brand">CloudVisionAI</span>
                <span class="cvai-workspace-badge">Workspace</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with ws_col_actions:
    ws_b1, ws_b2 = st.columns([1.4, 1])
    with ws_b1:
        if st.button("← Back to Landing", key="ws_back_landing_btn", use_container_width=True):
            st.session_state["active_view"] = "landing"
            st.rerun()
    with ws_b2:
        ws_theme_label = "☀️ Light" if get_theme() == "dark" else "🌙 Dark"
        if st.button(ws_theme_label, key="ws_theme_btn", use_container_width=True):
            toggle_theme()
            st.rerun()

st.title("🛰️ Satellite Cloud Removal & Reconstruction")
st.caption("Copernicus Data Space Ecosystem → Spatial Harmonization → Cloud Removal → Quantitative Evaluation & OpenSR")

tab_former, tab_cnn, tab_comparison = st.tabs(["Former Model", "CNN Model", "Comparison"])
with tab_former:
    tab_live, tab_cached = st.tabs(["Live Acquisition", "Cached Phase 2 Baseline"])
    # TAB 2: Cached Phase 2 Baseline (Validated Workflow)
    # ===========================================================================
    with tab_cached:
        st.header("Pre-Validated Phase 2 Harmonized Data")
        st.caption("Run canonical ECRformer on verified Phase 2 GeoTIFFs (EPSG:32633, 262x200 10m grid) or custom uploaded inputs.")
    
        data_mode = st.radio(
            "Input Source",
            ["Cached Phase 2 CDSE Data", "Upload Custom Harmonized GeoTIFFs"],
            horizontal=True,
            key="c_tab_mode",
        )
    
        s1_input_c = s2_input_c = mask_input_c = None
        input_ready_c = False
    
        if data_mode == "Cached Phase 2 CDSE Data":
            if DEFAULT_S1.exists() and DEFAULT_S2.exists() and DEFAULT_MASK.exists():
                s1_input_c = DEFAULT_S1
                s2_input_c = DEFAULT_S2
                mask_input_c = DEFAULT_MASK
                input_ready_c = True
                st.success("Pre-loaded verified Phase 2 harmonized pair: Rome Area (EPSG:32633, 262x200, 10m spatial resolution).")
            else:
                st.error("Cached harmonized data not found in harmonization/output/.")
        else:
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                sf1 = st.file_uploader("S1 SAR (2-band TIFF)", type=["tif", "tiff"], key="up_s1")
            with cc2:
                sf2 = st.file_uploader("S2 Optical (13-band TIFF)", type=["tif", "tiff"], key="up_s2")
            with cc3:
                sfm = st.file_uploader("Validity Mask (optional)", type=["tif", "tiff"], key="up_mask")
            if sf1 and sf2:
                s1_input_c = sf1.read()
                s2_input_c = sf2.read()
                mask_input_c = sfm.read() if sfm else None
                input_ready_c = True
    
        if input_ready_c:
            st.subheader("Input Previews")
            pc1, pc2, pc3 = st.columns(3)
            try:
                if isinstance(s1_input_c, Path):
                    with rasterio.open(s1_input_c) as src:
                        s1_arr = src.read()
                    with rasterio.open(s2_input_c) as src:
                        s2_arr = src.read()
                    with rasterio.open(mask_input_c) as src:
                        mask_arr = src.read(1).astype(bool)
                else:
                    with rasterio.io.MemoryFile(s1_input_c) as mf:
                        with mf.open() as src:
                            s1_arr = src.read()
                    with rasterio.io.MemoryFile(s2_input_c) as mf:
                        with mf.open() as src:
                            s2_arr = src.read()
                    mask_arr = None
                    if mask_input_c:
                        with rasterio.io.MemoryFile(mask_input_c) as mf:
                            with mf.open() as src:
                                mask_arr = src.read(1).astype(bool)
    
                with pc1:
                    st.image(sar_preview(s1_arr, valid_mask=mask_arr), caption="S1 SAR (VV/VH False Color)", use_container_width=True)
                with pc2:
                    st.image(to_rgb(s2_arr, brightness=brightness, valid_mask=mask_arr), caption="S2 Cloudy Optical (RGB)", use_container_width=True)
                with pc3:
                    if mask_arr is not None:
                        st.image(Image.fromarray((mask_arr.astype(np.uint8) * 255), mode="L"),
                                 caption=f"Spatial Validity Mask ({mask_arr.sum():,} valid px)", use_container_width=True)
            except Exception as e:
                st.warning(f"Preview rendering notice: {e}")
    
            st.divider()
            if st.button("Run ECRformer Cloud Removal", type="primary", use_container_width=True, key="cached_btn"):
                _config = InferenceConfig(tile_size=tile_size, overlap=overlap, brightness=brightness)
                pb = st.progress(0, text="Initializing canonical inference...")
    
                def _cb_c(done, total):
                    pb.progress(int(100 * done / max(total, 1)), text=f"Processing tiles: {done}/{total}")
    
                try:
                    result_c = service.run(
                        s1_input=s1_input_c,
                        s2_cloudy_input=s2_input_c,
                        validity_mask_input=mask_input_c,
                        config=_config,
                        progress_callback=_cb_c,
                    )
                    pb.progress(100, text="Inference Complete!")
                    st.session_state["cached_ecr_result"] = result_c
                    st.session_state.pop("cached_sr_result", None)
                    try:
                        record_run(
                            run_id=f"cached_{uuid.uuid4().hex[:8]}",
                            mode="cached_baseline",
                            aoi_label="Rome Area (EPSG:32633)",
                            s2_date="Cached S2",
                            s1_date="Cached S1",
                            crs="EPSG:32633",
                            dimensions=f"{result_c.diagnostics.output_shape[2]}x{result_c.diagnostics.output_shape[1]}",
                            execution_time_s=result_c.diagnostics.execution_time_seconds,
                            device=result_c.diagnostics.device,
                            ecrformer_output_path=str(DEFAULT_S2) if DEFAULT_S2.exists() else None,
                        )
                    except Exception:
                        pass
                except Exception as exc:
                    st.error(f"Inference failed: {exc}")
    
            cached_res: Optional[InferenceResult] = st.session_state.get("cached_ecr_result")
            if cached_res is not None:
                d = cached_res.diagnostics
                st.success(f"ECRformer Reconstruction Successful ({d.execution_time_seconds:.2f}s on {d.device})")
    
                r1, r2 = st.columns(2)
                with r1:
                    st.image(cached_res.cloudy_rgb, caption="Input Cloudy S2", use_container_width=True)
                with r2:
                    st.image(cached_res.prediction_rgb, caption="ECRformer Reconstructed Cloud-Free (13 bands | 10 m)", use_container_width=True)
    
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Device", d.device.split(":")[0].upper())
                m2.metric("Time", f"{d.execution_time_seconds:.2f}s")
                m3.metric("Tiles", d.tile_count)
                m4.metric("Valid px", f"{d.valid_pixels:,}")
                m5.metric("Mean Refl.", f"{d.mean_val:.4f}")
    
                dc1, dc2, dc3 = st.columns(3)
                with dc1:
                    st.download_button(
                        "Download 13-Band GeoTIFF",
                        data=cached_res.to_geotiff_bytes(),
                        file_name="s2_cloud_free_13band.tif",
                        mime="image/tiff",
                        use_container_width=True,
                        key="c_dl_tif",
                    )
                with dc2:
                    st.download_button(
                        "Download RGB GeoTIFF (10 m)",
                        data=cached_res.to_rgb_geotiff_bytes(),
                        file_name="s2_cloud_free_rgb_10m.tif",
                        mime="image/tiff",
                        use_container_width=True,
                        key="c_dl_rgb_tif",
                        help="Three-band B04/B03/B02 GeoTIFF on the reconstructed 10 m grid, scaled 0-10000 to match the super-resolution product.",
                    )
                with dc3:
                    st.download_button(
                        "Download Preview PNG",
                        data=cached_res.to_png_bytes(),
                        file_name="s2_cloud_free_preview.png",
                        mime="image/png",
                        use_container_width=True,
                        key="c_dl_png",
                    )
    
                # Dedicated ECRformer Metrics & Diagnostics Area
                render_ecrformer_metrics_ui(
                    result=cached_res,
                    image_id="cached_baseline",
                    key_prefix="c_ecr",
                    allow_reference_upload=True,
                )
    
                # Optional OpenSR Super-Resolution Stage
                st.divider()
                st.markdown("### Optional Post-Processing: Diffusion Super-Resolution (OpenSR 4×)")
                st.caption("Apply OpenSR LDSR-S2 4× diffusion super-resolution to bands B04, B03, B02, B08 (10 m -> 2.5 m resolution).")
                st.info("ℹ️ **OpenSR output: B04/B03/B02/B08 only.** The 13-band ECRformer GeoTIFF above remains the authoritative full-spectrum scientific reconstruction product.")
    
                cached_sr_model = st.selectbox(
                    "Diffusion checkpoint",
                    options=("v1", "v2-preview"),
                    format_func=lambda version: (
                        "LDSR-S2 v1 (recommended)" if version == "v1"
                        else "LDSR-S2 v2 research preview (experimental)"
                    ),
                    key="cached_sr_model_version",
                    help="V1 remains the canonical checkpoint. V2 is an experimental preview and is not established as a quality upgrade.",
                )
                if cached_sr_model == "v2-preview":
                    st.caption("V2 preview uses DPM-Solver++ (3 steps) and back-projection. The larger full checkpoint is for training-state recovery, not inference.")
    
                if st.button("✨ Generate Super-Resolved RGBN (OpenSR 4×)", key="cached_btn_sr", use_container_width=True):
                    with st.spinner(f"Executing OpenSR LDSR-S2 {cached_sr_model} on the available device..."):
                        try:
                            temp_dir = ROOT / "reconstructed_images" / "sessions" / "cached_baseline"
                            temp_dir.mkdir(parents=True, exist_ok=True)
                            temp_input = temp_dir / "ecrformer_13band_10m.tif"
                            cached_res.save_geotiff(temp_input)
    
                            sr_svc = OpenSRService(OpenSRConfig(model_version=cached_sr_model))
                            sr_res = sr_svc.run(input_tif_path=temp_input, output_dir=temp_dir)
                            st.session_state["cached_sr_result"] = sr_res
                            st.success("OpenSR Super-Resolution completed successfully!")
                        except Exception as exc:
                            st.error(f"OpenSR Super-Resolution failed: {exc}")
    
                cached_sr: Optional[OpenSRResult] = st.session_state.get("cached_sr_result")
                if cached_sr is not None:
                    st.subheader("OpenSR Super-Resolution (4 bands | 2.5 m | 4×)")
    
                    sr_v1, sr_v2 = st.columns(2)
                    with sr_v1:
                        st.image(cached_res.prediction_rgb, caption="ECRformer 10 m Baseline (RGB)", use_container_width=True)
                    with sr_v2:
                        st.image(cached_sr.sr_rgb_preview, caption="OpenSR 2.5 m Super-Resolved (RGB: B04, B03, B02)", use_container_width=True)
    
                    srdl1, srdl2 = st.columns(2)
                    with srdl1:
                        st.download_button(
                            "Download 4-Band SR GeoTIFF (2.5m)",
                            data=cached_sr.to_geotiff_bytes(),
                            file_name="s2_sr_4band_2.5m.tif",
                            mime="image/tiff",
                            use_container_width=True,
                            key="c_sr_dl_tif",
                        )
                    with srdl2:
                        st.download_button(
                            "Download 2.5m RGB Preview PNG",
                            data=cached_sr.to_png_bytes(),
                            file_name="s2_sr_2.5m_preview.png",
                            mime="image/png",
                            use_container_width=True,
                            key="c_sr_dl_png",
                        )
    
                    # Dedicated OpenSR Metrics & Diagnostics Area
                    render_opensr_metrics_ui(
                        sr_res=cached_sr,
                        image_id="cached_baseline",
                        key_prefix="c_sr",
                    )
    # ===========================================================================
    # TAB 1: Live Acquisition Workflow
    # ===========================================================================
    with tab_live:
        st.header("Live CDSE Acquisition Workflow")
        st.caption("Search a location or landmark, specify AOI extent, query CDSE Sentinel collections, harmonize, and reconstruct.")
    
        if not (os.environ.get("CDSE_CLIENT_ID") and os.environ.get("CDSE_CLIENT_SECRET")):
            st.info("💡 **Copernicus CDSE Credentials:** Live catalogue search requires CDSE credentials. Configure them in the **left sidebar** under **CDSE Authentication**, or test with the cached Rome benchmark below.")
    
        # ---- Canonical State Initialization ----
        if "selected_bbox" not in st.session_state:
            # Default: Rome benchmark (cached data verified)
            st.session_state["selected_bbox"] = BoundingBox(west=12.4500, south=41.9000, east=12.4730, north=41.9230)
        if "resolved_location" not in st.session_state:
            st.session_state["resolved_location"] = GeocodedLocation(
                query="Rome, Italy",
                display_name="Rome, Lazio, Italy (Benchmark Cache)",
                latitude=41.9115,
                longitude=12.4615,
            )
        if "aoi_size_km" not in st.session_state:
            st.session_state["aoi_size_km"] = 2.0
        if "aoi_source" not in st.session_state:
            st.session_state["aoi_source"] = "Default Benchmark (Rome)"
        if "map_center" not in st.session_state:
            st.session_state["map_center"] = (41.9115, 12.4615)
    
        cur_bbox: BoundingBox = st.session_state["selected_bbox"]
        cur_loc: Optional[GeocodedLocation] = st.session_state.get("resolved_location")
    
        # ------------------------------------------------------------------
        # STEP 1: Location Search & AOI Selection
        # ------------------------------------------------------------------
        with st.expander("Step 1 - Location Search & Area of Interest (AOI)", expanded=True):
            st.subheader("1. Search Location, City, Address or Landmark")
    
            search_col1, search_col2 = st.columns([4, 1])
            with search_col1:
                search_query = st.text_input(
                    "Search query",
                    value="",
                    placeholder="e.g. Mumbai, India | Airoli, Navi Mumbai | Neral, Maharashtra | Rome, Italy | Gateway of India",
                    label_visibility="collapsed",
                    key="loc_search_input",
                )
            with search_col2:
                search_submitted = st.button("Search Location", type="primary", use_container_width=True, key="btn_do_search")
    
            if search_submitted:
                if search_query.strip():
                    with st.spinner(f"Geocoding '{search_query.strip()}'..."):
                        loc = geocode_location(search_query.strip())
                    if loc is not None:
                        st.session_state["resolved_location"] = loc
                        st.session_state["map_center"] = (loc.latitude, loc.longitude)
                        st.session_state["selected_bbox"] = bbox_from_center(
                            loc.latitude, loc.longitude, st.session_state["aoi_size_km"]
                        )
                        st.session_state["aoi_source"] = f"Geocoded: {loc.query}"
                        st.rerun()
                    else:
                        st.error(f"Could not find location for '{search_query}'. Please check the query or enter coordinates manually.")
                else:
                    st.warning("Please enter a location name, address, or landmark.")
    
            # Quick Preset Buttons
            st.caption("Quick Location Presets:")
            pcol1, pcol2, pcol3, pcol4, pcol5, pcol6 = st.columns(6)
            with pcol1:
                if st.button("Mumbai, India", key="p_mum", use_container_width=True):
                    loc = geocode_location("Mumbai, India")
                    if loc:
                        st.session_state["resolved_location"] = loc
                        st.session_state["map_center"] = (loc.latitude, loc.longitude)
                        st.session_state["selected_bbox"] = bbox_from_center(loc.latitude, loc.longitude, st.session_state["aoi_size_km"])
                        st.session_state["aoi_source"] = "Preset: Mumbai, India"
                        st.rerun()
            with pcol2:
                if st.button("Airoli, Navi Mumbai", key="p_air", use_container_width=True):
                    loc = geocode_location("Airoli, Navi Mumbai")
                    if loc:
                        st.session_state["resolved_location"] = loc
                        st.session_state["map_center"] = (loc.latitude, loc.longitude)
                        st.session_state["selected_bbox"] = bbox_from_center(loc.latitude, loc.longitude, st.session_state["aoi_size_km"])
                        st.session_state["aoi_source"] = "Preset: Airoli, Navi Mumbai"
                        st.rerun()
            with pcol3:
                if st.button("Neral, MH", key="p_ner", use_container_width=True):
                    loc = geocode_location("Neral, Maharashtra")
                    if loc:
                        st.session_state["resolved_location"] = loc
                        st.session_state["map_center"] = (loc.latitude, loc.longitude)
                        st.session_state["selected_bbox"] = bbox_from_center(loc.latitude, loc.longitude, st.session_state["aoi_size_km"])
                        st.session_state["aoi_source"] = "Preset: Neral, Maharashtra"
                        st.rerun()
            with pcol4:
                if st.button("Gateway of India", key="p_gat", use_container_width=True):
                    loc = geocode_location("Gateway of India")
                    if loc:
                        st.session_state["resolved_location"] = loc
                        st.session_state["map_center"] = (loc.latitude, loc.longitude)
                        st.session_state["selected_bbox"] = bbox_from_center(loc.latitude, loc.longitude, st.session_state["aoi_size_km"])
                        st.session_state["aoi_source"] = "Preset: Gateway of India"
                        st.rerun()
            with pcol5:
                if st.button("Rome (Cached)", key="p_rom", use_container_width=True):
                    st.session_state["selected_bbox"] = BoundingBox(west=12.4500, south=41.9000, east=12.4730, north=41.9230)
                    st.session_state["resolved_location"] = GeocodedLocation(
                        query="Rome, Italy",
                        display_name="Rome Benchmark, Lazio, Italy (Cache Verified)",
                        latitude=41.9115,
                        longitude=12.4615,
                    )
                    st.session_state["map_center"] = (41.9115, 12.4615)
                    st.session_state["aoi_source"] = "Preset: Rome Benchmark"
                    st.rerun()
            with pcol6:
                if st.button("Po Valley", key="p_pov", use_container_width=True):
                    st.session_state["selected_bbox"] = BoundingBox(west=10.5000, south=44.8000, east=10.5300, north=44.8300)
                    st.session_state["resolved_location"] = GeocodedLocation(
                        query="Po Valley Benchmark",
                        display_name="Po Valley Agricultural Benchmark, Italy",
                        latitude=44.8150,
                        longitude=10.5150,
                    )
                    st.session_state["map_center"] = (44.8150, 10.5150)
                    st.session_state["aoi_source"] = "Preset: Po Valley Benchmark"
                    st.rerun()
    
            st.divider()
    
            # ---- AOI Size Selection ----
            st.subheader("2. Choose AOI Dimension")
            size_c1, size_c2 = st.columns([3, 2])
    
            size_presets = {
                "1 km": 1.0,
                "2 km (Standard)": 2.0,
                "5 km": 5.0,
                "10 km": 10.0,
                "Custom": None,
            }
    
            cur_sz = st.session_state.get("aoi_size_km", 2.0)
            default_idx = 1
            if abs(cur_sz - 1.0) < 0.1:
                default_idx = 0
            elif abs(cur_sz - 2.0) < 0.1:
                default_idx = 1
            elif abs(cur_sz - 5.0) < 0.1:
                default_idx = 2
            elif abs(cur_sz - 10.0) < 0.1:
                default_idx = 3
            else:
                default_idx = 4
    
            with size_c1:
                size_choice = st.radio(
                    "AOI Extent Size Preset",
                    list(size_presets.keys()),
                    index=default_idx,
                    horizontal=True,
                    key="radio_aoi_size",
                )
    
            new_size_km = cur_sz
            with size_c2:
                if size_choice == "Custom":
                    new_size_km = st.number_input(
                        "Custom Size (km)",
                        min_value=0.5,
                        max_value=50.0,
                        value=float(cur_sz),
                        step=0.5,
                        key="custom_size_val",
                    )
                else:
                    new_size_km = size_presets[size_choice]
    
            if abs(new_size_km - cur_sz) > 1e-4:
                st.session_state["aoi_size_km"] = new_size_km
                center_lat, center_lon = st.session_state["map_center"]
                st.session_state["selected_bbox"] = bbox_from_center(center_lat, center_lon, new_size_km)
                st.rerun()
    
            cur_bbox = st.session_state["selected_bbox"]
            cur_loc = st.session_state.get("resolved_location")
            center_lat = (cur_bbox.south + cur_bbox.north) / 2.0
            center_lon = (cur_bbox.west + cur_bbox.east) / 2.0
            approx_w_km = cur_bbox.approx_width_m() / 1000.0
            approx_h_km = cur_bbox.approx_height_m() / 1000.0
    
            # ---- Active AOI Summary Info Box ----
            info_c1, info_c2 = st.columns([3, 2])
            with info_c1:
                loc_label = cur_loc.display_name if cur_loc else "Custom Region"
                st.markdown(f"**Target Location:** {loc_label}")
                st.markdown(f"**Center Coordinates:** `{center_lat:.5f}° N, {center_lon:.5f}° E`")
                st.markdown(f"**Extent Dimension:** `~{approx_w_km:.2f} km × {approx_h_km:.2f} km` (Source: *{st.session_state['aoi_source']}*)")
            with info_c2:
                st.markdown(
                    f"**Bounding Box (WGS84):**\n"
                    f"- West: `{cur_bbox.west:.5f}` | East: `{cur_bbox.east:.5f}`\n"
                    f"- South: `{cur_bbox.south:.5f}` | North: `{cur_bbox.north:.5f}`"
                )
    
            st.divider()
            st.subheader("3. Interactive Map")
            st.caption("Map view of the active acquisition area and center pin.")
            try:
                import folium
                from streamlit_folium import st_folium

                if max(approx_w_km, approx_h_km) <= 1.5:
                    zoom_lvl = 15
                elif max(approx_w_km, approx_h_km) <= 3.0:
                    zoom_lvl = 14
                elif max(approx_w_km, approx_h_km) <= 7.0:
                    zoom_lvl = 13
                elif max(approx_w_km, approx_h_km) <= 15.0:
                    zoom_lvl = 12
                else:
                    zoom_lvl = 11

                f_map = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_lvl)
                folium.Marker(
                    location=[center_lat, center_lon],
                    popup=cur_loc.display_name if cur_loc else "AOI Center",
                    tooltip="AOI Center",
                    icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
                ).add_to(f_map)
                folium.Rectangle(
                    bounds=[[cur_bbox.south, cur_bbox.west], [cur_bbox.north, cur_bbox.east]],
                    color="#FF3333",
                    fill=True,
                    fill_opacity=0.18,
                    weight=2,
                    tooltip=f"Active AOI (~{approx_w_km:.2f} km x {approx_h_km:.2f} km)",
                ).add_to(f_map)
                st_folium(f_map, width=1100, height=390, key="aoi_folium_map")
            except Exception as e:
                st.warning(f"Map component notice: {e}")

            output_px = st.select_slider(
                "Acquisition Download Grid",
                options=[128, 256, 512],
                value=256,
                key="slider_raw_grid_px",
            )
            st.caption("Raw acquisition dimension. Downstream spatial harmonization resamples to the authoritative 10m UTM grid.")
    
            # ---- Final AOI Validation ----
            bbox_valid, bbox_err = validate_aoi(cur_bbox)
            if not bbox_valid:
                st.error(f"AOI Validation Error: {bbox_err}")
                bbox = None
            else:
                bbox = cur_bbox
                output_size_px = (output_px, output_px)
                st.success(
                    f"**Active AOI Confirmed:** W={bbox.west:.4f}, S={bbox.south:.4f}, E={bbox.east:.4f}, N={bbox.north:.4f} | "
                    f"~{approx_w_km:.2f} km × {approx_h_km:.2f} km | Download Grid: {output_px}×{output_px} px"
                )
    
        # ------------------------------------------------------------------
        # STEP 2: Date Range
        # ------------------------------------------------------------------
        with st.expander("Step 2 - Date Range", expanded=True):
            d1, d2 = st.columns(2)
            with d1:
                start_date = st.date_input("Start Date", value=datetime.date(2024, 9, 1), key="start_date")
            with d2:
                end_date = st.date_input("End Date", value=datetime.date(2024, 9, 30), key="end_date")
    
            if start_date >= end_date:
                st.error("Start date must be strictly before end date.")
                date_valid = False
                start_str = end_str = ""
            else:
                date_valid = True
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")
        # ------------------------------------------------------------------
        # STEP 3: CDSE Catalogue Search
        # ------------------------------------------------------------------
        with st.expander("Step 3 - Search CDSE Catalogue", expanded=True):
            search_disabled = not (bbox_valid and date_valid)
    
            btn_c1, btn_c2 = st.columns(2)
            with btn_c1:
                search_btn = st.button("Search CDSE Catalogue (Live)", disabled=search_disabled, key="search_btn", type="primary")
            with btn_c2:
                offline_btn = st.button("Load Cached Rome Scenes (Offline Demo)", key="offline_btn")
    
            if search_btn and bbox_valid and date_valid:
                for k in ["s2_scenes", "s1_scenes", "selected_s2_idx", "selected_s1_idx", "catalogue_provider",
                          "s2_sel", "s1_sel", "former_auto_s1_for_s2_idx"]:
                    st.session_state.pop(k, None)
    
                try:
                    from backend.services.satellite.cdse_sentinelhub import CDSESentinelHubProvider
                    provider = CDSESentinelHubProvider()
                    st.session_state["catalogue_provider"] = provider
                except Exception as exc:
                    st.error(f"Authentication error: {exc}. Please configure CDSE credentials in the sidebar or use the Offline Demo.")
                    provider = None
    
                if provider is not None:
                    with st.spinner("Searching Sentinel-2 L1C catalogue..."):
                        try:
                            st.session_state["s2_scenes"] = provider.search_sentinel2(
                                bbox=bbox, start_date=start_str, end_date=end_str, max_results=25,
                            )
                        except Exception as exc:
                            st.error(f"Sentinel-2 search failed: {exc}")
                            st.session_state["s2_scenes"] = []
    
                    with st.spinner("Searching Sentinel-1 GRD catalogue..."):
                        try:
                            st.session_state["s1_scenes"] = provider.search_sentinel1(
                                bbox=bbox, start_date=start_str, end_date=end_str, max_results=25,
                            )
                        except Exception as exc:
                            st.error(f"Sentinel-1 search failed: {exc}")
                            st.session_state["s1_scenes"] = []
    
            if offline_btn:
                st.session_state.pop("former_auto_s1_for_s2_idx", None)
                st.session_state.pop("s1_sel", None)
                st.session_state.pop("s2_sel", None)
                st.session_state["selected_bbox"] = BoundingBox(west=12.4500, south=41.9000, east=12.4730, north=41.9230)
                st.session_state["resolved_location"] = GeocodedLocation(
                    query="Rome, Italy",
                    display_name="Rome Benchmark, Lazio, Italy (Cache Verified)",
                    latitude=41.9115,
                    longitude=12.4615,
                )
                st.session_state["map_center"] = (41.9115, 12.4615)
                st.session_state["aoi_source"] = "Preset: Rome Benchmark (Cached)"
                st.session_state["s2_scenes"] = [
                    S2Scene(
                        scene_id="S2A_MSIL1C_20240921T100031_N0511_R122_T32TQM_20240921T120138.SAFE",
                        dt=datetime.datetime(2024, 9, 21, 10, 0, 31, tzinfo=datetime.timezone.utc),
                        cloud_cover=4.26,
                        processing_level="L1C",
                        footprint=None,
                    )
                ]
                st.session_state["s1_scenes"] = [
                    S1Scene(
                        scene_id="S1A_IW_GRDH_1SDV_20240920T051201_20240920T051226_055744_06CF2B_B9A8_COG.SAFE",
                        dt=datetime.datetime(2024, 9, 20, 5, 12, 1, tzinfo=datetime.timezone.utc),
                        product_type="GRD",
                        polarizations=["VV", "VH"],
                        orbit_direction="DESCENDING",
                        instrument_mode="IW",
                        footprint=None,
                    )
                ]
                st.session_state["selected_s2_idx"] = 0
                st.session_state["selected_s1_idx"] = 0
                st.success("Loaded verified Rome cached acquisition metadata and set Rome AOI.")
                st.rerun()
    
            # Display S2 Catalogue Results
            if "s2_scenes" in st.session_state:
                s2_scenes = st.session_state["s2_scenes"]
                st.subheader(f"Sentinel-2 L1C Results ({len(s2_scenes)} scenes)")
                if not s2_scenes:
                    st.warning("No Sentinel-2 scenes found for this AOI and date range.")
                else:
                    s2_df = pd.DataFrame([{
                        "#": i + 1,
                        "Date": sc.utc_date(),
                        "Time UTC": sc.utc_time(),
                        "Cloud %": f"{sc.cloud_cover:.1f}%" if sc.cloud_cover is not None else "N/A",
                        "Scene ID": sc.scene_id,
                    } for i, sc in enumerate(s2_scenes)])
                    st.dataframe(s2_df, use_container_width=True, hide_index=True)
    
                    s2_opts = {
                        f"[{i + 1}] {sc.utc_date()} | Cloud: {sc.cloud_str().strip()} | {sc.scene_id[:45]}": i
                        for i, sc in enumerate(s2_scenes)
                    }
                    cur_s2_idx = st.session_state.get("selected_s2_idx")
                    chosen_s2 = st.selectbox(
                        "Select Sentinel-2 scene",
                        list(s2_opts.keys()),
                        index=cur_s2_idx if (cur_s2_idx is not None and cur_s2_idx < len(s2_opts)) else None,
                        placeholder="-- Choose an S2 scene --",
                        key="s2_sel",
                    )
                    if chosen_s2 is not None:
                        st.session_state["selected_s2_idx"] = s2_opts[chosen_s2]

            selected_s2_idx_for_match = st.session_state.get("selected_s2_idx")
            s2_for_match = st.session_state.get("s2_scenes", [])
            s1_for_match = st.session_state.get("s1_scenes", [])
            st.session_state.pop("s1_sel", None)  # Clear stale state from the old manual selector.
            auto_s1_idx = (
                best_s1_scene_index(s2_for_match[selected_s2_idx_for_match], s1_for_match)
                if selected_s2_idx_for_match is not None and selected_s2_idx_for_match < len(s2_for_match)
                else None
            )
            st.session_state["selected_s1_idx"] = auto_s1_idx
            if auto_s1_idx is not None:
                auto_scene = s1_for_match[auto_s1_idx]
                gap = abs((auto_scene.dt - s2_for_match[selected_s2_idx_for_match].dt).total_seconds()) / 86400.0
                st.info(f"Automatically selected closest Sentinel-1 GRD: {gap:.2f} days from S2 · {auto_scene.pol_str()}.")
            elif s1_for_match:
                st.warning("Select a Sentinel-2 scene to automatically match its closest Sentinel-1 GRD scene.")
    
            # Display S1 Catalogue Results
            if "s1_scenes" in st.session_state:
                s1_scenes = st.session_state["s1_scenes"]
                st.subheader(f"Sentinel-1 GRD Results ({len(s1_scenes)} scenes)")
                if not s1_scenes:
                    st.warning("No Sentinel-1 scenes found for this AOI and date range.")
                else:
                    s1_df = pd.DataFrame([{
                        "#": i + 1,
                        "Date": sc.utc_date(),
                        "Time UTC": sc.utc_time(),
                        "Polarization": sc.pol_str(),
                        "Orbit": sc.orbit_direction or "?",
                        "Mode": sc.instrument_mode or "?",
                        "VV+VH": "Yes" if sc.has_vv_vh() else "No",
                        "Scene ID": sc.scene_id,
                    } for i, sc in enumerate(s1_scenes)])
                    st.dataframe(s1_df, use_container_width=True, hide_index=True)
    
                    if auto_s1_idx is None:
                        st.warning("No Sentinel-1 scene can be matched until a Sentinel-2 scene is selected.")
                    else:
                        st.caption(f"Scene {auto_s1_idx + 1} is selected automatically based on the shortest time gap.")
    
        # ------------------------------------------------------------------
        # STEP 4: Confirm Selection & Temporal Gap
        # ------------------------------------------------------------------
        with st.expander("Step 4 - Confirm Scene Selection & Temporal Alignment", expanded=True):
            sel_s2_idx = st.session_state.get("selected_s2_idx")
            sel_s1_idx = st.session_state.get("selected_s1_idx")
            s2_scenes_all = st.session_state.get("s2_scenes", [])
            s1_scenes_all = st.session_state.get("s1_scenes", [])
    
            sel_s2 = s2_scenes_all[sel_s2_idx] if (sel_s2_idx is not None and sel_s2_idx < len(s2_scenes_all)) else None
            sel_s1 = s1_scenes_all[sel_s1_idx] if (sel_s1_idx is not None and sel_s1_idx < len(s1_scenes_all)) else None
    
            cc1, cc2 = st.columns(2)
            with cc1:
                if sel_s2:
                    st.success(
                        f"**Selected S2:** {sel_s2.utc_date()} {sel_s2.utc_time()} UTC | "
                        f"Cloud: {sel_s2.cloud_str().strip()} | {sel_s2.scene_id[:35]}..."
                    )
                else:
                    st.info("No Sentinel-2 scene selected.")
            with cc2:
                if sel_s1:
                    st.success(
                        f"**Selected S1:** {sel_s1.utc_date()} {sel_s1.utc_time()} UTC | "
                        f"{sel_s1.pol_str()} {sel_s1.orbit_direction or '?'} | {sel_s1.scene_id[:35]}..."
                    )
                else:
                    st.info("No Sentinel-1 scene selected.")
    
            both_selected = (sel_s2 is not None) and (sel_s1 is not None)
    
            if both_selected:
                delta_sec = abs((sel_s2.dt - sel_s1.dt).total_seconds())
                delta_days = int(delta_sec // 86400)
                delta_hrs = int((delta_sec % 86400) // 3600)
                s2_dt_str = sel_s2.dt.strftime('%Y-%m-%d %H:%M')
                s1_dt_str = sel_s1.dt.strftime('%Y-%m-%d %H:%M')
                gap_text = f"S2: {s2_dt_str} UTC | S1: {s1_dt_str} UTC | Temporal gap: **{delta_days}d {delta_hrs}h**"
    
                if delta_days > 7:
                    st.warning(f"{gap_text} | Warning: Temporal gap exceeds 7 days. Surface changes may be present.")
                elif delta_days > 3:
                    st.info(f"{gap_text} | Temporal gap is between 3 and 7 days. Acceptable for most terrain.")
                else:
                    st.success(f"{gap_text} | Excellent temporal alignment (within 3 days)!")
            else:
                st.caption("Select both an S2 scene and an S1 scene above to evaluate temporal gap and enable acquisition.")
    
        # ------------------------------------------------------------------
        # STEP 5: Acquire, Harmonize & Run ECRformer
        # ------------------------------------------------------------------
        with st.expander("Step 5 - Acquire, Harmonize & Run ECRformer", expanded=True):
            acquire_btn = st.button(
                "Acquire + Harmonize + Run ECRformer",
                type="primary",
                disabled=not both_selected,
                use_container_width=True,
                key="acquire_btn",
            )
    
            if acquire_btn and both_selected:
                session_id = str(uuid.uuid4())[:8]
                session_harm_dir = HARM_OUTPUT_BASE / "sessions" / session_id
                session_harm_dir.mkdir(parents=True, exist_ok=True)
    
                log_lines = []
                log_box = st.empty()
    
                def log(msg, ok=True):
                    icon = "[OK]" if ok else "[ERR]"
                    log_lines.append(f"{icon} {msg}")
                    log_box.code("\n".join(log_lines))
    
                # --- S2 Acquisition ---
                s2_cache_dir = get_s2_cache_dir(CACHE_DIR, bbox, sel_s2.utc_date(), output_size_px)
                s2_cache_dir.mkdir(parents=True, exist_ok=True)
                s2_tif = s2_cache_dir / "s2.tif"
                s2_ok = False
    
                if is_s2_cached(CACHE_DIR, bbox, sel_s2.utc_date(), output_size_px):
                    log(f"Sentinel-2 optical acquired from cache ({s2_tif.stat().st_size / 1024:.1f} KB)")
                    s2_ok = True
                else:
                    provider = st.session_state.get("catalogue_provider")
                    if provider is None:
                        try:
                            from backend.services.satellite.cdse_sentinelhub import CDSESentinelHubProvider
                            provider = CDSESentinelHubProvider()
                            st.session_state["catalogue_provider"] = provider
                        except Exception as exc:
                            st.error(f"CDSE credentials required for download: {exc}")
                            log("Sentinel-2 download FAILED - credentials not configured", ok=False)
    
                    if provider is not None:
                        log(f"Downloading Sentinel-2 ({sel_s2.utc_date()}) from CDSE...")
                        try:
                            provider.download_sentinel2(
                                bbox=bbox, target_date=sel_s2.utc_date(),
                                output_path=str(s2_tif), output_size_px=output_size_px,
                            )
                            write_cache_metadata(
                                s2_cache_dir, "sentinel-2-l1c", bbox, sel_s2.utc_date(),
                                S2_BAND_NAMES, output_size_px,
                                extra={"scene_id": sel_s2.scene_id, "cloud_cover": sel_s2.cloud_cover},
                            )
                            log(f"Sentinel-2 downloaded successfully ({s2_tif.stat().st_size / 1024:.1f} KB)")
                            s2_ok = True
                        except Exception as exc:
                            st.error(f"Sentinel-2 acquisition failed: {exc}")
                            log(f"Sentinel-2 download FAILED: {exc}", ok=False)
    
                # --- S1 Acquisition ---
                s1_ok = False
                if s2_ok:
                    s1_cache_dir = get_s1_cache_dir(CACHE_DIR, bbox, sel_s1.utc_date(), output_size_px)
                    s1_cache_dir.mkdir(parents=True, exist_ok=True)
                    s1_tif = s1_cache_dir / "s1.tif"
    
                    if is_s1_cached(CACHE_DIR, bbox, sel_s1.utc_date(), output_size_px):
                        log(f"Sentinel-1 SAR acquired from cache ({s1_tif.stat().st_size / 1024:.1f} KB)")
                        s1_ok = True
                    else:
                        provider = st.session_state.get("catalogue_provider")
                        if provider is None:
                            try:
                                from backend.services.satellite.cdse_sentinelhub import CDSESentinelHubProvider
                                provider = CDSESentinelHubProvider()
                                st.session_state["catalogue_provider"] = provider
                            except Exception as exc:
                                st.error(f"CDSE credentials required for download: {exc}")
                                log("Sentinel-1 download FAILED - credentials not configured", ok=False)
    
                        if provider is not None:
                            log(f"Downloading Sentinel-1 ({sel_s1.utc_date()}) from CDSE...")
                            try:
                                provider.download_sentinel1(
                                    bbox=bbox, target_date=sel_s1.utc_date(),
                                    output_path=str(s1_tif), output_size_px=output_size_px,
                                )
                                write_cache_metadata(
                                    s1_cache_dir, "sentinel-1-grd", bbox, sel_s1.utc_date(),
                                    S1_BAND_NAMES, output_size_px,
                                    extra={
                                        "scene_id": sel_s1.scene_id,
                                        "polarizations": sel_s1.polarizations,
                                        "orbit_direction": sel_s1.orbit_direction,
                                        "instrument_mode": sel_s1.instrument_mode,
                                    },
                                )
                                log(f"Sentinel-1 downloaded successfully ({s1_tif.stat().st_size / 1024:.1f} KB)")
                                s1_ok = True
                            except Exception as exc:
                                st.error(f"Sentinel-1 acquisition failed: {exc}")
                                log(f"Sentinel-1 download FAILED: {exc}", ok=False)
    
                # --- Phase 2 Spatial Harmonization ---
                harm_ok = False
                harm_pair = None
                if s1_ok and s2_ok:
                    log("Executing Phase 2 spatial harmonization (UTM reprojection, 10m grid, validity mask)...")
                    try:
                        h_config = HarmonizationConfig(
                            pixel_size=10.0,
                            s2_resampling=RioRes.bilinear,
                            s1_resampling=RioRes.bilinear,
                            save_previews=True,
                            save_validity_mask=True,
                        )
                        harm_pair = harmonize_pair(
                            s2_input_path=s2_tif,
                            s1_input_path=s1_tif,
                            output_dir=str(session_harm_dir),
                            config=h_config,
                        )
                        log(f"Harmonization completed: {harm_pair.s2_shape[2]}x{harm_pair.s2_shape[1]} px, EPSG:{harm_pair.grid.crs.to_epsg()}")
                        harm_ok = True
                    except Exception as exc:
                        st.error(f"Harmonization failed: {exc}")
                        log(f"Harmonization FAILED: {exc}", ok=False)
    
                # --- Canonical ECRformer Inference ---
                if harm_ok and (harm_pair is not None):
                    log("Executing ECRformer deep reconstruction on harmonized inputs...")
                    try:
                        infer_cfg = InferenceConfig(tile_size=tile_size, overlap=overlap, brightness=brightness)
                        result = service.run(
                            s1_input=harm_pair.s1_path,
                            s2_cloudy_input=harm_pair.s2_path,
                            validity_mask_input=harm_pair.validity_mask_path,
                            config=infer_cfg,
                        )
                        d = result.diagnostics
                        log(f"ECRformer reconstruction succeeded in {d.execution_time_seconds:.2f}s on {d.device}!")
                        log(f"Diagnostics: {d.tile_count} tiles, {d.valid_pixels:,} valid pixels, mean={d.mean_val:.4f}")
                        st.session_state["live_result"] = result
                        st.session_state["live_harm_pair"] = harm_pair
                        st.session_state["live_s2_meta"] = sel_s2
                        st.session_state["live_bbox"] = bbox
                        st.session_state.pop("live_sr_result", None)
                        st.session_state.pop("live_clear_ref", None)
                        st.session_state.pop("live_clear_ref_candidates", None)
                        st.session_state.pop("live_clear_ref_cand_meta", None)
                        try:
                            date_val = sel_s2.utc_date() if sel_s2 else "live"
                            s1_val = sel_s1.utc_date() if (sel_s1 is not None and hasattr(sel_s1, "utc_date")) else None
                            record_run(
                                run_id=f"live_{uuid.uuid4().hex[:8]}",
                                mode="live_acquisition",
                                aoi_label=loc_name if ("loc_name" in locals() and loc_name) else "Live AOI",
                                s2_date=date_val,
                                s1_date=s1_val,
                                crs=str(harm_pair.crs) if harm_pair else None,
                                dimensions=f"{d.output_shape[2]}x{d.output_shape[1]}",
                                execution_time_s=d.execution_time_seconds,
                                device=d.device,
                                ecrformer_output_path=str(harm_pair.s2_path) if harm_pair else None,
                            )
                        except Exception:
                            pass
                    except Exception as exc:
                        st.error(f"ECRformer inference failed: {exc}")
                        log(f"Inference FAILED: {exc}", ok=False)
    
        # ------------------------------------------------------------------
        # RESULTS & VISUALIZATION (Tab 1)
        # ------------------------------------------------------------------
        live_res: Optional[InferenceResult] = st.session_state.get("live_result")
        live_harm: Optional[HarmonizedPair] = st.session_state.get("live_harm_pair")
        live_s2 = st.session_state.get("live_s2_meta")
    
        if live_res is not None:
            st.divider()
            st.subheader("Reconstruction Results")
            diag = live_res.diagnostics
            st.success(f"ECRformer Reconstruction Successful ({diag.execution_time_seconds:.2f}s on {diag.device})")
    
            # Metric cards
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Compute Device", diag.device.split(":")[0].upper())
            m2.metric("Inference Time", f"{diag.execution_time_seconds:.2f}s")
            m3.metric("Tiles Processed", diag.tile_count)
            m4.metric("Valid Pixels", f"{diag.valid_pixels:,}")
            m5.metric("Mean Reflectance", f"{diag.mean_val:.4f}")
    
            # Visual Comparison (3 Columns)
            v1, v2, v3 = st.columns(3)
            with v1:
                st.image(live_res.cloudy_rgb, caption="Input Sentinel-2 Cloudy (RGB)", use_container_width=True)
            with v2:
                st.image(live_res.sar_preview_img, caption="Harmonized Sentinel-1 SAR (VV/VH False Color)", use_container_width=True)
            with v3:
                st.image(live_res.prediction_rgb, caption="ECRformer Reconstructed Cloud-Free (RGB)", use_container_width=True)
    
            if live_harm and live_harm.preview_coreg_path and live_harm.preview_coreg_path.exists():
                with st.expander("SAR / Optical Co-Registration Diagnostics", expanded=False):
                    st.image(str(live_harm.preview_coreg_path), caption="SAR High-Frequency Edge Overlay (Magenta) on Optical S2", use_container_width=True)
    
            # Downloads
            st.subheader("Export & Download")
            dl1, dl2, dl3, dl4 = st.columns(4)
            date_str = live_s2.utc_date() if live_s2 else "output"
    
            with dl1:
                st.download_button(
                    "Download 13-Band GeoTIFF",
                    data=live_res.to_geotiff_bytes(),
                    file_name=f"s2_reconstructed_{date_str}_13band.tif",
                    mime="image/tiff",
                    use_container_width=True,
                    key="live_dl_tif",
                )
            with dl2:
                st.download_button(
                    "Download RGB GeoTIFF (10 m)",
                    data=live_res.to_rgb_geotiff_bytes(),
                    file_name=f"s2_reconstructed_{date_str}_rgb_10m.tif",
                    mime="image/tiff",
                    use_container_width=True,
                    key="live_dl_rgb_tif",
                    help="Three-band B04/B03/B02 GeoTIFF on the reconstructed 10 m grid, scaled 0-10000 to match the super-resolution product.",
                )
            with dl3:
                st.download_button(
                    "Download Preview PNG",
                    data=live_res.to_png_bytes(),
                    file_name=f"s2_reconstructed_{date_str}_preview.png",
                    mime="image/png",
                    use_container_width=True,
                    key="live_dl_png",
                )
            with dl4:
                diag_json = json.dumps(diag.to_dict(), indent=2)
                st.download_button(
                    "Download Diagnostics JSON",
                    data=diag_json,
                    file_name="diagnostics.json",
                    mime="application/json",
                    use_container_width=True,
                    key="live_dl_json",
                )
    
            # Dedicated ECRformer Metrics & Diagnostics Area
            render_ecrformer_metrics_ui(
                result=live_res,
                image_id=f"session_{date_str}",
                key_prefix="live_ecr",
                allow_reference_upload=True,
            )
    
            # --------------------------------------------------------------
            # Live Clear-Reference Search (CDSE Sentinel-2)
            # --------------------------------------------------------------
            active_bbox = st.session_state.get("live_bbox") or bbox
            target_date_str = live_s2.utc_date() if live_s2 else None
    
            if active_bbox is not None and target_date_str is not None:
                with st.expander("🔍 Find a Clear Reference for This AOI (CDSE Sentinel-2)", expanded=False):
                    st.caption(
                        "Search the CDSE catalogue for low-cloud Sentinel-2 scenes covering this exact AOI "
                        "within a configurable temporal window. A selected candidate will be acquired, geospatially "
                        "aligned to the ECRformer prediction grid, and compared without modifying inference inputs."
                    )
    
                    rc1, rc2, rc3 = st.columns(3)
                    with rc1:
                        window_days = st.slider(
                            "Search Window (±Days)",
                            min_value=5,
                            max_value=90,
                            value=30,
                            step=5,
                            key="live_ref_window",
                        )
                    with rc2:
                        max_scene_cloud = st.slider(
                            "Max Scene Cloud %",
                            min_value=1.0,
                            max_value=50.0,
                            value=15.0,
                            step=1.0,
                            key="live_ref_max_scene_cloud",
                        )
                    with rc3:
                        max_aoi_cloud = st.slider(
                            "Max AOI Cloud %",
                            min_value=1.0,
                            max_value=30.0,
                            value=10.0,
                            step=1.0,
                            key="live_ref_max_aoi_cloud",
                        )
    
                    rc4, rc5 = st.columns(2)
                    with rc4:
                        ref_collection = st.selectbox(
                            "Sentinel-2 Collection",
                            ["sentinel-2-l2a", "sentinel-2-l1c"],
                            index=0,
                            key="live_ref_collection",
                            help="Level-2A provides Bottom-of-Atmosphere reflectance and Scene Classification (SCL). Level-1C provides Top-of-Atmosphere reflectance.",
                        )
                    with rc5:
                        max_cands = st.number_input(
                            "Max Candidates",
                            min_value=1,
                            max_value=20,
                            value=6,
                            key="live_ref_max_cands",
                        )
    
                    if st.button("Search CDSE for Clear Reference Scenes", key="btn_search_clear_ref", use_container_width=True):
                        with st.spinner(f"Querying CDSE catalogue for clear scenes around {target_date_str} (±{window_days} days)..."):
                            try:
                                ref_provider = st.session_state.get("catalogue_provider")
                                ref_svc = ClearReferenceService(provider=ref_provider)
                                candidates = ref_svc.search_candidates(
                                    bbox=active_bbox,
                                    target_date=target_date_str,
                                    window_days=int(window_days),
                                    max_scene_cloud_cover=float(max_scene_cloud),
                                    max_aoi_cloud_fraction=float(max_aoi_cloud),
                                    max_candidates=int(max_cands),
                                    collection=ref_collection,
                                )
                                st.session_state["live_clear_ref_candidates"] = candidates
                                if not candidates:
                                    st.info(
                                        f"No suitable clear reference scenes were found within ±{window_days} days "
                                        f"(scene cloud <= {max_scene_cloud}%, AOI cloud <= {max_aoi_cloud}%). "
                                        "Try expanding the date window or cloud thresholds, or upload a manual reference GeoTIFF above."
                                    )
                            except Exception as e:
                                st.error(f"CDSE catalogue search failed: {e}")
    
                    candidates_list = st.session_state.get("live_clear_ref_candidates", [])
                    if candidates_list:
                        st.markdown("##### Candidate Clear Scenes Found in CDSE")
                        table_rows = []
                        for c in candidates_list:
                            sign = "+" if c.temporal_distance_days >= 0 else ""
                            table_rows.append({
                                "Date": c.date_str,
                                "Temporal Offset": f"{sign}{c.temporal_distance_days:.1f} days",
                                "Scene Cloud %": f"{c.scene_cloud_cover_pct:.1f}%" if c.scene_cloud_cover_pct is not None else "N/A",
                                "AOI Cloud %": f"{c.aoi_cloud_fraction_pct:.1f}%" if c.aoi_cloud_fraction_pct is not None else "N/A (needs download)",
                                "Status": "✅ " + c.status if c.passes_threshold else "⚠️ " + c.status,
                                "Product ID": (c.scene_id[:35] + "...") if len(c.scene_id) > 35 else c.scene_id,
                            })
                        st.dataframe(table_rows, use_container_width=True)
    
                        cand_labels = [c.display_label() for c in candidates_list]
                        sel_cand_idx = st.selectbox(
                            "Select Candidate Reference Scene to Download & Inspect",
                            range(len(candidates_list)),
                            format_func=lambda i: cand_labels[i],
                            key="sel_clear_ref_cand",
                        )
                        selected_cand = candidates_list[sel_cand_idx]
    
                        if st.button("📥 Download, Align & Inspect Candidate Reference", key="btn_download_clear_ref", type="primary", use_container_width=True):
                            with st.spinner(f"Downloading and aligning Sentinel-2 scene from {selected_cand.date_str}..."):
                                try:
                                    ref_provider = st.session_state.get("catalogue_provider")
                                    ref_svc = ClearReferenceService(provider=ref_provider)
                                    tgt_w = live_res.profile.get("width", 256)
                                    tgt_h = live_res.profile.get("height", 256)
                                    val_ref = ref_svc.acquire_and_align_reference(
                                        candidate=selected_cand,
                                        bbox=active_bbox,
                                        target_profile=live_res.profile,
                                        scale_mode="auto",
                                        output_size_px=(tgt_w, tgt_h),
                                    )
                                    st.session_state["live_clear_ref"] = val_ref
                                    st.session_state["live_clear_ref_cand_meta"] = selected_cand
                                    st.success(f"Clear reference from {selected_cand.date_str} successfully acquired and geospatially aligned!")
                                except Exception as e:
                                    st.error(f"Failed to acquire reference: {e}")
    
                    val_ref = st.session_state.get("live_clear_ref")
                    sel_cand_meta = st.session_state.get("live_clear_ref_cand_meta")
    
                    if val_ref is not None and sel_cand_meta is not None:
                        st.divider()
                        st.markdown("#### Reference Inspection & Comparison")
                        ref_rgb = ClearReferenceService.to_rgb_preview(val_ref)
    
                        c_ref1, c_ref2 = st.columns(2)
                        with c_ref1:
                            sign = "+" if sel_cand_meta.temporal_distance_days >= 0 else ""
                            st.image(
                                ref_rgb,
                                caption=f"Clear Reference ({val_ref.acquisition_date}, {sign}{val_ref.temporal_distance_days:.1f}d)",
                                use_container_width=True,
                            )
                        with c_ref2:
                            st.image(
                                live_res.prediction_rgb,
                                caption=f"ECRformer Reconstructed Output ({target_date_str})",
                                use_container_width=True,
                            )
    
                        st.warning(
                            "⚠️ **This is a clear temporal reference from a different acquisition date. "
                            "Reference-based differences may reflect real surface or atmospheric changes and "
                            "should not be interpreted as strict reconstruction accuracy.**"
                        )
    
                        r_info1, r_info2, r_info3 = st.columns(3)
                        r_info1.metric("Reference Date", val_ref.acquisition_date or "N/A")
                        sign = "+" if sel_cand_meta.temporal_distance_days >= 0 else ""
                        r_info2.metric("Temporal Distance", f"{sign}{sel_cand_meta.temporal_distance_days:.1f} days")
                        r_info3.metric("Grid Alignment", f"{val_ref.array.shape[2]}×{val_ref.array.shape[1]} (EPSG:{val_ref.profile.get('crs')})")
    
                        if st.button("📊 Calculate Reference-Based Metrics (Temporal)", key="btn_eval_live_clear_ref", type="secondary", use_container_width=True):
                            with st.spinner("Computing 13-band temporal reference comparison metrics..."):
                                try:
                                    metrics = live_res.evaluate_with_reference(val_ref)
                                    st.session_state["live_ecr_ref_metrics"] = metrics
                                    st.success("Temporal reference comparison metrics computed successfully!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Evaluation error: {e}")
    
            # Optional OpenSR Super-Resolution Stage (Live)
            st.divider()
            st.subheader("Optional Post-Processing: Diffusion Super-Resolution (OpenSR 4×)")
            st.caption("Apply OpenSR LDSR-S2 4× diffusion super-resolution to bands B04, B03, B02, B08 (10 m -> 2.5 m resolution).")
            st.info("ℹ️ **OpenSR output: B04/B03/B02/B08 only.** The 13-band ECRformer GeoTIFF above remains the authoritative full-spectrum scientific reconstruction product.")
    
            live_sr_model = st.selectbox(
                "Diffusion checkpoint",
                options=("v1", "v2-preview"),
                format_func=lambda version: (
                    "LDSR-S2 v1 (recommended)" if version == "v1"
                    else "LDSR-S2 v2 research preview (experimental)"
                ),
                key="live_sr_model_version",
                help="V1 remains the canonical checkpoint. V2 is an experimental preview and is not established as a quality upgrade.",
            )
            if live_sr_model == "v2-preview":
                st.caption("V2 preview uses DPM-Solver++ (3 steps) and back-projection. The larger full checkpoint is for training-state recovery, not inference.")
    
            if st.button("✨ Generate Super-Resolved RGBN (OpenSR 4×)", key="live_btn_sr", use_container_width=True):
                with st.spinner(f"Executing OpenSR LDSR-S2 {live_sr_model} on the available device..."):
                    try:
                        live_out_dir = ROOT / "reconstructed_images" / "sessions" / f"session_{date_str}"
                        live_out_dir.mkdir(parents=True, exist_ok=True)
                        live_ecr_tif = live_out_dir / f"s2_ecrformer_{date_str}_13band.tif"
                        live_res.save_geotiff(live_ecr_tif)
    
                        sr_svc = OpenSRService(OpenSRConfig(model_version=live_sr_model))
                        sr_res = sr_svc.run(input_tif_path=live_ecr_tif, output_dir=live_out_dir)
                        st.session_state["live_sr_result"] = sr_res
                        st.success("OpenSR Super-Resolution completed successfully!")
                    except Exception as exc:
                        st.error(f"OpenSR Super-Resolution failed: {exc}")
    
            live_sr: Optional[OpenSRResult] = st.session_state.get("live_sr_result")
            if live_sr is not None:
                st.subheader("OpenSR Super-Resolution (4 bands | 2.5 m | 4×)")
    
                sr_lv1, sr_lv2 = st.columns(2)
                with sr_lv1:
                    st.image(live_res.prediction_rgb, caption="ECRformer 10 m Baseline (RGB)", use_container_width=True)
                with sr_lv2:
                    st.image(live_sr.sr_rgb_preview, caption="OpenSR 2.5 m Super-Resolved (RGB: B04, B03, B02)", use_container_width=True)
    
                lsrdl1, lsrdl2 = st.columns(2)
                with lsrdl1:
                    st.download_button(
                        "Download 4-Band SR GeoTIFF (2.5m)",
                        data=live_sr.to_geotiff_bytes(),
                        file_name=f"s2_sr_{date_str}_4band_2.5m.tif",
                        mime="image/tiff",
                        use_container_width=True,
                        key="live_sr_dl_tif",
                    )
                with lsrdl2:
                    st.download_button(
                        "Download 2.5m RGB Preview PNG",
                        data=live_sr.to_png_bytes(),
                        file_name=f"s2_sr_{date_str}_preview.png",
                        mime="image/png",
                        use_container_width=True,
                        key="live_sr_dl_png",
                    )
    
                # Dedicated OpenSR Metrics & Diagnostics Area
                render_opensr_metrics_ui(
                    sr_res=live_sr,
                    image_id=f"session_{date_str}",
                    key_prefix="live_sr",
                )


with tab_cnn:
    render_cnn_workspace(
        brightness=brightness,
        service_getter=get_dsen2cr_service,
        acquire_and_harmonize=acquire_and_harmonize_for_cnn,
        model_input_selector=render_model_input_pair_selector,
        run_inference=run_cnn_inference,
        render_result=render_cnn_result,
    )

with tab_comparison:
    st.header("Former Model vs CNN Comparison")
    st.caption("Choose one input source. A single run processes that same pair with Former first, then CNN.")

    compare_source_mode = st.radio(
        "Input source",
        ["CDSE harmonized data", "Upload GeoTIFFs"],
        horizontal=True,
        key="comparison_input_source_mode",
    )
    compare_s1 = compare_s2 = compare_mask = None
    compare_ground_truth = None
    compare_source = ""
    compare_token = None
    uploaded_pair = False

    if compare_source_mode == "CDSE harmonized data":
        if "comparison_cdse_bbox" not in st.session_state:
            st.session_state["comparison_cdse_bbox"] = BoundingBox(12.4500, 41.9000, 12.4730, 41.9230)
        comp_bbox = st.session_state["comparison_cdse_bbox"]
        st.session_state.setdefault("comparison_cdse_center_lat", (comp_bbox.south + comp_bbox.north) / 2.0)
        st.session_state.setdefault("comparison_cdse_center_lon", (comp_bbox.west + comp_bbox.east) / 2.0)
        st.session_state.setdefault("comparison_cdse_aoi_size_km", 2.0)
        st.session_state.setdefault("comparison_cdse_location_label", "Comparison AOI")
        for comp_key, comp_value in (("comparison_cdse_west", comp_bbox.west), ("comparison_cdse_south", comp_bbox.south),
                                     ("comparison_cdse_east", comp_bbox.east), ("comparison_cdse_north", comp_bbox.north)):
            st.session_state.setdefault(comp_key, float(comp_value))

        with st.expander("Step 1 · Location and Area of Interest", expanded=True):
            location_cols = st.columns([4, 1])
            with location_cols[0]:
                location_query = st.text_input("Search location", key="comparison_cdse_location_query", placeholder="City, address, or landmark")
            with location_cols[1]:
                if st.button("Search", key="comparison_cdse_location_search", use_container_width=True):
                    found_location = geocode_location(location_query)
                    if found_location:
                        st.session_state["comparison_cdse_center_lat"] = found_location.latitude
                        st.session_state["comparison_cdse_center_lon"] = found_location.longitude
                        st.session_state["comparison_cdse_location_label"] = found_location.display_name
                        new_bbox = bbox_from_center(found_location.latitude, found_location.longitude,
                                                    float(st.session_state["comparison_cdse_aoi_size_km"]))
                        st.session_state["comparison_cdse_bbox"] = new_bbox
                        for comp_key, comp_value in (("comparison_cdse_west", new_bbox.west), ("comparison_cdse_south", new_bbox.south),
                                                     ("comparison_cdse_east", new_bbox.east), ("comparison_cdse_north", new_bbox.north)):
                            st.session_state[comp_key] = float(comp_value)
                        st.session_state.pop("comparison_live_harm_pair", None)
                        st.rerun()
                    else:
                        st.error("Location was not found. Try another search.")
            st.caption(f"Current AOI: {st.session_state['comparison_cdse_location_label']}")
            center_cols = st.columns(3)
            with center_cols[0]:
                st.number_input("Center latitude", min_value=-90.0, max_value=90.0, format="%.5f", key="comparison_cdse_center_lat")
            with center_cols[1]:
                st.number_input("Center longitude", min_value=-180.0, max_value=180.0, format="%.5f", key="comparison_cdse_center_lon")
            with center_cols[2]:
                st.slider("AOI size (km)", min_value=0.5, max_value=20.0, step=0.5, key="comparison_cdse_aoi_size_km")
            if st.button("Set AOI from center and size", key="comparison_cdse_center_apply", use_container_width=True):
                new_bbox = bbox_from_center(
                    float(st.session_state["comparison_cdse_center_lat"]),
                    float(st.session_state["comparison_cdse_center_lon"]),
                    float(st.session_state["comparison_cdse_aoi_size_km"]),
                )
                st.session_state["comparison_cdse_bbox"] = new_bbox
                st.session_state["comparison_cdse_location_label"] = "Center and Size AOI"
                for comp_key, comp_value in (("comparison_cdse_west", new_bbox.west), ("comparison_cdse_south", new_bbox.south),
                                             ("comparison_cdse_east", new_bbox.east), ("comparison_cdse_north", new_bbox.north)):
                    st.session_state[comp_key] = float(comp_value)
                st.session_state.pop("comparison_live_harm_pair", None)
                st.rerun()
            bounds_cols = st.columns(4)
            for col, label, key in zip(bounds_cols, ("West", "South", "East", "North"),
                                       ("comparison_cdse_west", "comparison_cdse_south", "comparison_cdse_east", "comparison_cdse_north")):
                with col:
                    st.number_input(label, format="%.5f", key=key)
            if st.button("Apply manual bounds", key="comparison_cdse_apply_bounds", use_container_width=True):
                manual_bbox = BoundingBox(
                    west=float(st.session_state["comparison_cdse_west"]),
                    south=float(st.session_state["comparison_cdse_south"]),
                    east=float(st.session_state["comparison_cdse_east"]),
                    north=float(st.session_state["comparison_cdse_north"]),
                )
                valid_aoi, aoi_error = validate_aoi(manual_bbox)
                if valid_aoi:
                    st.session_state["comparison_cdse_bbox"] = manual_bbox
                    st.session_state["comparison_cdse_location_label"] = "Manual Coordinates"
                    st.session_state.pop("comparison_live_harm_pair", None)
                    st.rerun()
                else:
                    st.error(aoi_error)
            valid_aoi, aoi_error = validate_aoi(st.session_state["comparison_cdse_bbox"])
            if valid_aoi:
                st.success("AOI is valid and ready for catalogue search.")
            else:
                st.error(aoi_error)

        today = datetime.date.today()
        with st.expander("Step 2 · Date Range", expanded=True):
            date_cols = st.columns(2)
            with date_cols[0]:
                comp_start = st.date_input("Start date", value=today - datetime.timedelta(days=30), key="comparison_cdse_start_date")
            with date_cols[1]:
                comp_end = st.date_input("End date", value=today, key="comparison_cdse_end_date")
            comp_dates_valid = comp_start < comp_end
            if not comp_dates_valid:
                st.error("Start date must be before end date.")

        with st.expander("Step 3 · Search CDSE Catalogue", expanded=True):
            grid_col, search_col = st.columns([1, 2])
            with grid_col:
                comp_grid_px = st.select_slider("Download grid", options=[128, 256, 512], value=256, key="comparison_cdse_grid_px")
            with search_col:
                comp_search = st.button("Search Sentinel-1 and Sentinel-2", type="primary", use_container_width=True,
                                        key="comparison_cdse_catalogue_search",
                                        disabled=not (comp_dates_valid and validate_aoi(st.session_state["comparison_cdse_bbox"])[0]))
            if comp_search:
                for comp_key in ("comparison_s2_scenes", "comparison_s1_scenes", "comparison_selected_s2_idx",
                                 "comparison_selected_s1_idx", "comparison_s2_scene_widget", "comparison_s1_scene_widget",
                                 "comparison_live_harm_pair", "comparison_auto_s1_for_s2_idx"):
                    st.session_state.pop(comp_key, None)
                try:
                    provider = st.session_state.get("comparison_catalogue_provider")
                    if provider is None:
                        from backend.services.satellite.cdse_sentinelhub import CDSESentinelHubProvider
                        provider = CDSESentinelHubProvider()
                        st.session_state["comparison_catalogue_provider"] = provider
                    with st.spinner("Searching the CDSE catalogue..."):
                        aoi = st.session_state["comparison_cdse_bbox"]
                        st.session_state["comparison_s2_scenes"] = provider.search_sentinel2(
                            bbox=aoi, start_date=comp_start.strftime("%Y-%m-%d"), end_date=comp_end.strftime("%Y-%m-%d"), max_results=25,
                        )
                        st.session_state["comparison_s1_scenes"] = provider.search_sentinel1(
                            bbox=aoi, start_date=comp_start.strftime("%Y-%m-%d"), end_date=comp_end.strftime("%Y-%m-%d"), max_results=25,
                        )
                    st.success("Catalogue search completed.")
                except Exception as exc:
                    st.error(f"CDSE catalogue search failed: {exc}")
                    st.session_state["comparison_s2_scenes"] = []
                    st.session_state["comparison_s1_scenes"] = []

            s2_scenes = st.session_state.get("comparison_s2_scenes", [])
            s1_scenes = st.session_state.get("comparison_s1_scenes", [])
            if s2_scenes:
                st.subheader(f"Sentinel-2 L1C scenes ({len(s2_scenes)})")
                s2_labels = [f"[{i + 1}] {scene.utc_date()} · Cloud {scene.cloud_str().strip()} · {scene.scene_id[:45]}" for i, scene in enumerate(s2_scenes)]
                st.dataframe(pd.DataFrame([{"Date": scene.utc_date(), "Time UTC": scene.utc_time(),
                                            "Cloud %": scene.cloud_str().strip(), "Scene ID": scene.scene_id}
                                           for scene in s2_scenes]), use_container_width=True, hide_index=True)
                prior = st.session_state.get("comparison_selected_s2_idx")
                choice = st.selectbox("Select Sentinel-2 scene", range(len(s2_scenes)),
                                      index=prior if prior is not None and prior < len(s2_scenes) else None,
                                      format_func=lambda i: s2_labels[i], key="comparison_s2_scene_widget")
                st.session_state["comparison_selected_s2_idx"] = choice
            selected_s2_idx_for_match = st.session_state.get("comparison_selected_s2_idx")
            st.session_state.pop("comparison_s1_scene_widget", None)  # Clear old manual-selector state.
            auto_s1_idx = (
                best_s1_scene_index(s2_scenes[selected_s2_idx_for_match], s1_scenes)
                if selected_s2_idx_for_match is not None and selected_s2_idx_for_match < len(s2_scenes)
                else None
            )
            st.session_state["comparison_selected_s1_idx"] = auto_s1_idx
            if auto_s1_idx is not None:
                auto_scene = s1_scenes[auto_s1_idx]
                gap = abs((auto_scene.dt - s2_scenes[selected_s2_idx_for_match].dt).total_seconds()) / 86400.0
                st.info(f"Automatically selected closest Sentinel-1 GRD: {gap:.2f} days from S2 · {auto_scene.pol_str()}.")
            elif s1_scenes:
                st.warning("Select a Sentinel-2 scene to automatically match its closest Sentinel-1 GRD scene.")
            if s1_scenes:
                st.subheader(f"Sentinel-1 GRD scenes ({len(s1_scenes)})")
                st.dataframe(pd.DataFrame([{"Date": scene.utc_date(), "Polarization": scene.pol_str(),
                                            "Orbit": scene.orbit_direction or "?", "Scene ID": scene.scene_id}
                                           for scene in s1_scenes]), use_container_width=True, hide_index=True)
                if auto_s1_idx is None:
                    st.warning("No Sentinel-1 scene can be matched until a Sentinel-2 scene is selected.")
                else:
                    st.caption(f"Scene {auto_s1_idx + 1} is selected automatically based on the shortest time gap.")

            selected_s2_idx = st.session_state.get("comparison_selected_s2_idx")
            selected_s1_idx = st.session_state.get("comparison_selected_s1_idx")
            scenes_ready = (selected_s2_idx is not None and selected_s1_idx is not None
                            and selected_s2_idx < len(s2_scenes) and selected_s1_idx < len(s1_scenes))
            if scenes_ready:
                selected_s2_scene = s2_scenes[selected_s2_idx]
                selected_s1_scene = s1_scenes[selected_s1_idx]
                gap_days = abs((selected_s2_scene.dt - selected_s1_scene.dt).total_seconds()) / 86400.0
                if gap_days > 7:
                    st.error(f"Selected scenes are {gap_days:.1f} days apart. Choose S1 within 7 days of S2.")
                elif gap_days > 3:
                    st.warning(f"Selected scenes are {gap_days:.1f} days apart; a closer pair is preferred.")
                else:
                    st.success(f"Selected scene gap: {gap_days:.1f} days.")
                comp_pair_usable = gap_days <= 7
                if st.button("Acquire and harmonize selected pair", key="comparison_cdse_acquire", type="primary",
                             use_container_width=True, disabled=not comp_pair_usable):
                    try:
                        with st.spinner("Acquiring scenes and harmonizing on the common 10 m grid..."):
                            comparison_pair = acquire_and_harmonize_selected_pair(
                                st.session_state["comparison_cdse_bbox"], selected_s2_scene, selected_s1_scene,
                                int(st.session_state.get("comparison_cdse_grid_px", 256)),
                                "comparison", "comparison_catalogue_provider",
                            )
                        st.session_state["comparison_live_harm_pair"] = comparison_pair
                        st.session_state["comparison_live_bbox"] = st.session_state["comparison_cdse_bbox"]
                        st.session_state["comparison_live_s2_scene"] = selected_s2_scene
                        st.success("CDSE pair acquired and harmonized.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"CDSE acquisition or harmonization failed: {exc}")

        live_pair = st.session_state.get("comparison_live_harm_pair")
        if live_pair is not None and Path(live_pair.s1_path).is_file() and Path(live_pair.s2_path).is_file():
            compare_s1 = Path(live_pair.s1_path)
            compare_s2 = Path(live_pair.s2_path)
            compare_mask = Path(live_pair.validity_mask_path) if live_pair.validity_mask_path else None
            live_scene = st.session_state.get("comparison_live_s2_scene")
            compare_source = f"New CDSE pair · {live_scene.utc_date() if live_scene else 'acquired'}"
            compare_token = f"CDSE|{compare_source}|{compare_s1.stat().st_mtime_ns}|{compare_s2.stat().st_mtime_ns}"
            st.success(f"Comparison input ready: {compare_source}")
    else:
        upload_cols = st.columns(3)
        with upload_cols[0]:
            s1_upload = st.file_uploader("SAR GeoTIFF (VV/VH)", type=["tif", "tiff"], key="comparison_s1_upload",
                                         help="Accepts a standard 2-band VV/VH TIFF or the SEN12MS-CR 256×256 SAR TIFF layout.")
        with upload_cols[1]:
            s2_upload = st.file_uploader("Cloudy Sentinel-2 GeoTIFF (13 bands)", type=["tif", "tiff"], key="comparison_s2_upload")
        with upload_cols[2]:
            ground_truth_upload = st.file_uploader(
                "Cloud-free ground truth (optional, 13-band Sentinel-2 GeoTIFF)",
                type=["tif", "tiff"], key="comparison_ground_truth_upload",
                help="Used only to score the two reconstructions. It may use a different grid; it will be aligned automatically.",
            )
        if s1_upload and s2_upload:
            uploaded_pair = True
            compare_s1 = s1_upload.getvalue()
            compare_s2 = s2_upload.getvalue()
            compare_mask = None
            compare_ground_truth = ground_truth_upload.getvalue() if ground_truth_upload else None
            compare_source = "Uploaded S1/S2 GeoTIFFs"
            sar_ready = True
            try:
                compare_s1, sar_repacked = normalize_sen12ms_sar_upload(compare_s1, compare_s2)
                if sar_repacked:
                    st.info("Recognized SEN12MS-CR SAR layout. VV/VH bands were unpacked and aligned to the uploaded 256×256 Sentinel-2 patch grid.")
            except Exception as exc:
                st.error(f"SAR upload could not be prepared: {exc}")
                sar_ready = False
            import hashlib
            digest = hashlib.sha256(compare_s1 + compare_s2).hexdigest()
            if sar_ready:
                compare_token = f"Upload|{digest}"
            try:
                if not sar_ready:
                    raise ValueError("Correct the SAR upload format before running comparison.")
                with rasterio.io.MemoryFile(compare_s1) as s1_mem, rasterio.io.MemoryFile(compare_s2) as s2_mem:
                    with s1_mem.open() as s1_src, s2_mem.open() as s2_src:
                        if s1_src.count != 2 or s2_src.count != 13:
                            raise ValueError("Upload a 2-band S1 TIFF and a 13-band S2 TIFF.")
                        if (s1_src.width, s1_src.height) != (s2_src.width, s2_src.height) or s1_src.crs != s2_src.crs:
                            raise ValueError("S1 and S2 must already be harmonized to the same dimensions and CRS.")
                        if any(abs(a - b) > 1e-4 for a, b in zip(s1_src.transform, s2_src.transform)):
                            raise ValueError("S1 and S2 must use the same geospatial transform.")
                st.success("Uploaded input pair is ready for both models.")
                if compare_ground_truth is not None:
                    with rasterio.io.MemoryFile(compare_ground_truth) as gt_mem:
                        with gt_mem.open() as gt_src:
                            if gt_src.count != 13:
                                raise ValueError("Ground truth must contain exactly 13 Sentinel-2 bands.")
                    st.success("Ground-truth TIFF is ready; it will be aligned to each model output for scoring.")
            except Exception as exc:
                st.error(f"Uploaded input validation failed: {exc}")
                compare_token = None
        else:
            st.caption("Upload a paired SAR and cloudy Sentinel-2 patch. SAR must provide VV/VH in dB; Sentinel-2 must contain 13 bands. SEN12MS-CR SAR patches are supported.")
            compare_ground_truth = None

    if compare_source_mode != "Upload GeoTIFFs":
        compare_ground_truth = None

    if st.session_state.get("comparison_pair_token") != compare_token:
        st.session_state["comparison_pair_token"] = compare_token
        st.session_state.pop("comparison_former_result", None)
        st.session_state.pop("comparison_cnn_result", None)
        st.session_state.pop("comparison_former_sr_result", None)
        st.session_state.pop("comparison_cnn_sr_result", None)
        st.session_state.pop("comparison_reference_metrics", None)
        st.session_state.pop("comparison_opensr_gt_metrics", None)
        st.session_state.pop("comparison_opensr_gt_metric_key", None)

    input_ready = compare_s1 is not None and compare_s2 is not None and compare_token is not None
    if st.button("Run Former, then CNN", type="primary", use_container_width=True, key="comparison_run_both", disabled=not input_ready):
        st.session_state.pop("comparison_former_sr_result", None)
        st.session_state.pop("comparison_cnn_sr_result", None)
        st.session_state.pop("comparison_sr_key", None)
        st.session_state.pop("comparison_reference_metrics", None)
        st.session_state.pop("comparison_opensr_gt_metrics", None)
        st.session_state.pop("comparison_opensr_gt_metric_key", None)
        former_error = cnn_error = None
        with tempfile.TemporaryDirectory(prefix="comparison_pair_") as temp_dir:
            temp_dir = Path(temp_dir)
            if uploaded_pair:
                s1_run_path = temp_dir / "s1_uploaded.tif"
                s2_run_path = temp_dir / "s2_uploaded.tif"
                s1_run_path.write_bytes(compare_s1)
                s2_run_path.write_bytes(compare_s2)
                mask_run_path = None
            else:
                s1_run_path, s2_run_path, mask_run_path = compare_s1, compare_s2, compare_mask

            with st.spinner("1 of 2 · Running Former Model (ECRformer)..."):
                try:
                    former_compare = service.run(
                        s1_input=s1_run_path,
                        s2_cloudy_input=s2_run_path,
                        validity_mask_input=mask_run_path,
                        config=InferenceConfig(tile_size=tile_size, overlap=overlap, brightness=brightness),
                    )
                    st.session_state["comparison_former_result"] = former_compare
                except Exception as exc:
                    former_error = exc

            with st.spinner("2 of 2 · Running CNN Model (DSen2-CR)..."):
                try:
                    cnn_compare = get_dsen2cr_service().run(
                        s1_input=s1_run_path,
                        s2_cloudy_input=s2_run_path,
                        validity_mask_input=mask_run_path,
                        brightness=brightness,
                    )
                    if hasattr(cnn_compare, "cloud_guided_prediction"):
                        cnn_compare.prediction, _ = cnn_compare.cloud_guided_prediction(0.4)
                        cnn_compare.prediction_rgb = to_rgb(
                            cnn_compare.prediction, brightness=brightness, valid_mask=cnn_compare.valid_mask,
                        )
                        cnn_compare.output_mode = "Cloud-guided blend"
                        cnn_compare.cloud_score_threshold = 0.4
                    st.session_state["comparison_cnn_result"] = cnn_compare
                except Exception as exc:
                    cnn_error = exc

        if former_error:
            st.error(f"Former Model failed: {former_error}")
        if cnn_error:
            st.error(f"CNN Model failed: {cnn_error}")
        if former_error is None and cnn_error is None:
            st.success("Both models finished on the same input pair.")

    former_compare: Optional[InferenceResult] = st.session_state.get("comparison_former_result")
    cnn_compare: Optional[DSen2CRResult] = st.session_state.get("comparison_cnn_result")
    if input_ready and (former_compare is not None or cnn_compare is not None):
        if uploaded_pair:
            with rasterio.io.MemoryFile(compare_s2) as s2_mem:
                with s2_mem.open() as source:
                    cloudy = to_rgb(source.read(), brightness=brightness)
        else:
            with rasterio.open(compare_s2) as source:
                cloudy = to_rgb(source.read(), brightness=brightness)
        compare_columns = st.columns(3)
        with compare_columns[0]:
            st.image(cloudy, caption="Shared cloudy S2 input", use_container_width=True)
        with compare_columns[1]:
            if former_compare is not None:
                st.image(former_compare.prediction_rgb, caption=f"Former Model · {former_compare.diagnostics.execution_time_seconds:.2f}s", use_container_width=True)
            else:
                st.info("Former Model result is unavailable.")
        with compare_columns[2]:
            if cnn_compare is not None:
                st.image(cnn_compare.prediction_rgb, caption=f"DSen2-CR CNN (cloud-guided) · {cnn_compare.diagnostics.execution_time_seconds:.2f}s", use_container_width=True)
            else:
                st.info("CNN result is unavailable.")

        if former_compare is not None and cnn_compare is not None:
            st.subheader("Run Summary")
            summary1, summary2, summary3 = st.columns(3)
            summary1.metric("ECRformer time", f"{former_compare.diagnostics.execution_time_seconds:.2f}s")
            summary2.metric("DSen2-CR time", f"{cnn_compare.diagnostics.execution_time_seconds:.2f}s")
            summary3.metric("Same input grid", f"{former_compare.prediction.shape[2]} × {former_compare.prediction.shape[1]}")
            st.caption("Inference runtime on the shared input; shorter bars mean faster reconstruction.")
            runtime_data = pd.DataFrame({
                "Model": ["Former Model", "CNN"],
                "Seconds": [
                    former_compare.diagnostics.execution_time_seconds,
                    cnn_compare.diagnostics.execution_time_seconds,
                ],
            }).set_index("Model")
            st.bar_chart(
                runtime_data,
                y_label="Seconds",
                height=220,
                alt="Inference runtime comparison for Former Model and CNN",
            )
            if compare_ground_truth is not None:
                import hashlib
                reference_token = hashlib.sha256(compare_ground_truth).hexdigest()
                if st.session_state.get("comparison_reference_token") != reference_token:
                    st.session_state["comparison_reference_token"] = reference_token
                    st.session_state.pop("comparison_reference_metrics", None)
                    st.session_state.pop("comparison_opensr_gt_metrics", None)
                    st.session_state.pop("comparison_opensr_gt_metric_key", None)
                st.subheader("Compare against cloud-free ground truth")
                try:
                    with rasterio.io.MemoryFile(compare_ground_truth) as gt_mem:
                        with gt_mem.open() as gt_src:
                            gt_rgb = to_rgb(gt_src.read(), brightness=brightness)
                    st.image(gt_rgb, caption="Uploaded cloud-free reference", use_container_width=True)
                except Exception as exc:
                    st.error(f"Could not preview ground truth: {exc}")
                if st.button("Calculate metrics for both models", key="comparison_eval_both", use_container_width=True):
                    metrics_by_model = {}
                    for model_name, model_result in (("Former Model", former_compare), ("CNN", cnn_compare)):
                        try:
                            metrics_by_model[model_name] = model_result.evaluate_with_reference(
                                compare_ground_truth, scale_mode="auto", reference_type="paired_ground_truth",
                            )
                        except Exception as exc:
                            metrics_by_model[model_name] = exc
                    st.session_state["comparison_reference_metrics"] = metrics_by_model
                metrics_by_model = st.session_state.get("comparison_reference_metrics", {})
                if metrics_by_model:
                    rows = []
                    for model_name in ("Former Model", "CNN"):
                        metric = metrics_by_model.get(model_name)
                        if isinstance(metric, Exception):
                            st.error(f"{model_name} evaluation failed: {metric}")
                            continue
                        if metric is not None:
                            rows.append({
                                "Model": model_name,
                                "MAE ↓": f"{metric.aggregate_mae:.5f}",
                                "RMSE ↓": f"{metric.aggregate_rmse:.5f}",
                                "PSNR (dB) ↑": f"{metric.aggregate_psnr_db:.2f}",
                                "SSIM ↑": f"{metric.ssim_rgb:.4f}" if metric.ssim_rgb is not None else "—",
                                "SAM (°) ↓": f"{metric.sam_degrees:.2f}",
                                "Pixels": f"{metric.num_evaluated_pixels:,}",
                            })
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                        render_comparison_metric_charts(metrics_by_model, "comparison_cloud_removal")
                else:
                    st.caption("Ground-truth metrics align the reference to each 10 m output grid. Lower MAE/RMSE/SAM and higher PSNR/SSIM indicate a closer match.")
            else:
                st.caption("No cloud-free ground truth is attached, so this is a visual comparison only; it does not imply a measured quality ranking.")

            st.divider()
            st.subheader("Super-resolve both results with OpenSR")
            st.caption("Runs 4× OpenSR on both cloud-removal outputs (B04/B03/B02/B08, 10 m to 2.5 m).")
            sr_version = st.selectbox("OpenSR checkpoint", ["v1", "v2-preview"], key="comparison_opensr_version")
            sr_key = f"{st.session_state.get('comparison_pair_token', '')}|{sr_version}"
            if st.button("Run OpenSR on both results", key="comparison_opensr_both", type="primary", use_container_width=True):
                import hashlib
                run_id = hashlib.sha256(sr_key.encode("utf-8")).hexdigest()[:16]
                sr_root = ROOT / "reconstructed_images" / "sessions" / "comparison" / run_id
                former_input = sr_root / "former_cloud_free_13band.tif"
                cnn_input = sr_root / "cnn_cloud_free_13band.tif"
                sr_svc = OpenSRService(OpenSRConfig(model_version=sr_version))
                st.session_state.pop("comparison_former_sr_result", None)
                st.session_state.pop("comparison_cnn_sr_result", None)
                st.session_state.pop("comparison_opensr_gt_metrics", None)
                st.session_state.pop("comparison_opensr_gt_metric_key", None)
                sr_errors = {}
                for index, (model_name, model_result, input_path, output_path, state_key) in enumerate((
                    ("Former Model", former_compare, former_input, sr_root / "former_opensr", "comparison_former_sr_result"),
                    ("CNN", cnn_compare, cnn_input, sr_root / "cnn_opensr", "comparison_cnn_sr_result"),
                ), start=1):
                    try:
                        model_result.save_geotiff(input_path)
                        with st.spinner(f"OpenSR {index} of 2 · {model_name} output..."):
                            st.session_state[state_key] = sr_svc.run(input_tif_path=input_path, output_dir=output_path)
                    except Exception as exc:
                        sr_errors[model_name] = str(exc)
                        st.session_state.pop(state_key, None)
                if st.session_state.get("comparison_former_sr_result") or st.session_state.get("comparison_cnn_sr_result"):
                    st.session_state["comparison_sr_key"] = sr_key
                if sr_errors:
                    for model_name, error in sr_errors.items():
                        st.error(f"{model_name} OpenSR failed: {error}")
                if not sr_errors:
                    st.success("Super-resolution completed for both model outputs.")

            if st.session_state.get("comparison_sr_key") == sr_key:
                former_sr = st.session_state.get("comparison_former_sr_result")
                cnn_sr = st.session_state.get("comparison_cnn_sr_result")
                sr_cols = st.columns(2)
                for col, model_name, sr_result in ((sr_cols[0], "Former Model", former_sr), (sr_cols[1], "CNN", cnn_sr)):
                    with col:
                        if sr_result is None:
                            st.info(f"{model_name} OpenSR result is unavailable. Check the error shown above.")
                            continue
                        st.image(sr_result.sr_rgb_preview, caption=f"{model_name} · OpenSR 4×", use_container_width=True)
                        st.download_button(
                            f"Download {model_name} 2.5 m GeoTIFF", data=sr_result.to_geotiff_bytes(),
                            file_name=f"comparison_{model_name.lower().replace(' ', '_')}_opensr_2p5m.tif",
                            mime="image/tiff", key=f"comparison_download_{model_name.lower().replace(' ', '_')}_sr_tif",
                        )
                        st.download_button(
                            f"Download {model_name} 2.5 m preview", data=sr_result.to_png_bytes(),
                            file_name=f"comparison_{model_name.lower().replace(' ', '_')}_opensr_preview.png",
                            mime="image/png", key=f"comparison_download_{model_name.lower().replace(' ', '_')}_sr_png",
                        )

                st.divider()
                st.subheader("Ground-truth metrics after super-resolution")
                if compare_ground_truth is None:
                    st.info("Upload a cloud-free 13-band ground-truth TIFF in the Upload GeoTIFFs option to calculate these metrics.")
                else:
                    import hashlib
                    reference_token = hashlib.sha256(compare_ground_truth).hexdigest()
                    metric_key = f"{sr_key}|{reference_token}"
                    st.caption(
                        "OpenSR produces 2.5 m pixels, while the uploaded Sentinel-2 reference is 10 m. "
                        "The 4-band OpenSR result is area-averaged back to the reference's 10 m grid before scoring; "
                        "these metrics measure agreement at 10 m, not native 2.5 m detail accuracy. "
                        "The native OpenSR raster has 4× the pixels per side (16× total pixels); both counts are shown below."
                    )
                    if st.session_state.get("comparison_opensr_gt_metric_key") != metric_key:
                        opensr_metrics = {}
                        with st.spinner("Calculating OpenSR metrics against cloud-free ground truth..."):
                            for model_name, sr_result in (("Former Model", former_sr), ("CNN", cnn_sr)):
                                if sr_result is None:
                                    continue
                                try:
                                    opensr_metrics[model_name] = evaluate_opensr_against_ground_truth(
                                        sr_result, compare_ground_truth, former_compare.profile,
                                    )
                                except Exception as exc:
                                    opensr_metrics[model_name] = exc
                        st.session_state["comparison_opensr_gt_metrics"] = opensr_metrics
                        st.session_state["comparison_opensr_gt_metric_key"] = metric_key
                    opensr_metrics = (
                        st.session_state.get("comparison_opensr_gt_metrics", {})
                        if st.session_state.get("comparison_opensr_gt_metric_key") == metric_key else {}
                    )
                    metric_rows = []
                    for model_name, sr_result in (("Former Model", former_sr), ("CNN", cnn_sr)):
                        metric = opensr_metrics.get(model_name)
                        if isinstance(metric, Exception):
                            st.error(f"{model_name} OpenSR ground-truth evaluation failed: {metric}")
                        elif metric is not None:
                            metric_rows.append({
                                "Model": model_name,
                                "Compared bands": "B04, B03, B02, B08",
                                "Native OpenSR grid (2.5 m)": f"{sr_result.diagnostics.output_width} × {sr_result.diagnostics.output_height}",
                                "Native SR pixels": f"{sr_result.diagnostics.output_width * sr_result.diagnostics.output_height:,}",
                                "Ground-truth grid (10 m)": f"{former_compare.prediction.shape[2]} × {former_compare.prediction.shape[1]}",
                                "MAE ↓": f"{metric.aggregate_mae:.5f}",
                                "RMSE ↓": f"{metric.aggregate_rmse:.5f}",
                                "PSNR (dB) ↑": f"{metric.aggregate_psnr_db:.2f}",
                                "SSIM (RGB) ↑": f"{metric.ssim_rgb:.4f}" if metric.ssim_rgb is not None else "—",
                                "SAM (°) ↓": f"{metric.sam_degrees:.2f}",
                                "Ground-truth comparison pixels": f"{metric.num_evaluated_pixels:,}",
                            })
                    if metric_rows:
                        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
                        render_comparison_metric_charts(opensr_metrics, "comparison_opensr")
