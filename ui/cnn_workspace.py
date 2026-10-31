"""Full DSen2-CR workspace, parallel to the Former Model workflow UI."""
from __future__ import annotations

import datetime
import tempfile
from pathlib import Path

import pandas as pd
import rasterio
import streamlit as st

from backend.services.satellite.base import BoundingBox, S1Scene, S2Scene, best_s1_scene_index
from backend.services.satellite.geocoding import bbox_from_center, geocode_location, validate_aoi
from backend.services.satellite.acquisition_cache import S1_BAND_NAMES, S2_BAND_NAMES
from visualization import sar_preview, to_rgb


def _run_and_render(
    *,
    s1_path,
    s2_path,
    mask_path,
    source_label,
    result_key,
    sr_key,
    image_id,
    brightness,
    run_inference,
    render_result,
    bbox=None,
    scene_date=None,
    harm_pair=None,
):
    result = run_inference(
        s1_path, s2_path, mask_path, result_key, sr_key, source_label, brightness,
        bbox=bbox, target_date=scene_date, harm_pair=harm_pair,
    )
    render_result(
        result=result,
        image_id=image_id,
        key_prefix=result_key,
        sr_state_key=sr_key,
        harm_pair=harm_pair,
        bbox=bbox,
        target_date=scene_date,
    )


def _render_scene_search(acquire_and_harmonize, service_getter, run_inference, render_result, brightness):
    if "cnn_selected_bbox" not in st.session_state:
        st.session_state["cnn_selected_bbox"] = BoundingBox(west=12.4500, south=41.9000, east=12.4730, north=41.9230)
    bbox = st.session_state["cnn_selected_bbox"]
    if "cnn_center_lat" not in st.session_state:
        st.session_state["cnn_center_lat"] = (bbox.south + bbox.north) / 2.0
    if "cnn_center_lon" not in st.session_state:
        st.session_state["cnn_center_lon"] = (bbox.west + bbox.east) / 2.0
    if "cnn_location_label" not in st.session_state:
        st.session_state["cnn_location_label"] = "Rome Benchmark, Lazio, Italy (Cache Verified)"

    st.session_state.setdefault("cnn_aoi_size_km", 2.0)
    st.session_state.setdefault("cnn_aoi_source", "Default Benchmark (Rome)")

    with st.expander("Step 1 - Location Search & Area of Interest (AOI)", expanded=True):
        st.subheader("1. Search Location, City, Address or Landmark")
        search_col1, search_col2 = st.columns([4, 1])
        with search_col1:
            search_query = st.text_input(
                "Search query", value="",
                placeholder="e.g. Mumbai, India | Airoli, Navi Mumbai | Neral, Maharashtra | Rome, Italy | Gateway of India",
                label_visibility="collapsed", key="cnn_location_query",
            )
        with search_col2:
            if st.button("Search Location", type="primary", use_container_width=True, key="cnn_location_search"):
                if search_query.strip():
                    with st.spinner(f"Geocoding '{search_query.strip()}'..."):
                        loc = geocode_location(search_query.strip())
                    if loc is not None:
                        st.session_state["cnn_location_label"] = loc.display_name
                        st.session_state["cnn_aoi_source"] = f"Geocoded: {loc.query}"
                        st.session_state["cnn_center_lat"] = loc.latitude
                        st.session_state["cnn_center_lon"] = loc.longitude
                        st.session_state["cnn_selected_bbox"] = bbox_from_center(
                            loc.latitude, loc.longitude, float(st.session_state["cnn_aoi_size_km"]),
                        )
                        st.rerun()
                    st.error(f"Could not find location for '{search_query}'. Try another search.")
                else:
                    st.warning("Please enter a location name, address, or landmark.")

        st.caption("Quick Location Presets:")
        pcols = st.columns(6)
        presets = (
            ("Mumbai, India", "Mumbai, India", 19.0760, 72.8777),
            ("Airoli, Navi Mumbai", "Airoli, Navi Mumbai", 19.1538, 72.9967),
            ("Neral, MH", "Neral, Maharashtra", 19.0247, 73.3169),
            ("Gateway of India", "Gateway of India", 18.9219, 72.8347),
            ("Rome (Cached)", None, 41.9115, 12.4615),
            ("Po Valley", None, 44.8150, 10.5150),
        )
        for index, (label, query, latitude, longitude) in enumerate(presets):
            with pcols[index]:
                if st.button(label, key=f"cnn_preset_{index}", use_container_width=True):
                    if query:
                        found = geocode_location(query)
                        if found:
                            latitude, longitude = found.latitude, found.longitude
                            st.session_state["cnn_location_label"] = found.display_name
                            st.session_state["cnn_aoi_source"] = f"Preset: {label}"
                        else:
                            st.error(f"Could not find {query}.")
                            continue
                    else:
                        st.session_state["cnn_location_label"] = "Rome Benchmark (Cached)" if index == 4 else "Po Valley Agricultural Benchmark, Italy"
                        st.session_state["cnn_aoi_source"] = "Preset: Rome Benchmark" if index == 4 else "Preset: Po Valley Benchmark"
                    st.session_state["cnn_center_lat"] = latitude
                    st.session_state["cnn_center_lon"] = longitude
                    if index == 4:
                        st.session_state["cnn_selected_bbox"] = BoundingBox(west=12.4500, south=41.9000, east=12.4730, north=41.9230)
                    elif index == 5:
                        st.session_state["cnn_selected_bbox"] = BoundingBox(west=10.5000, south=44.8000, east=10.5300, north=44.8300)
                    else:
                        st.session_state["cnn_selected_bbox"] = bbox_from_center(
                            latitude, longitude, float(st.session_state["cnn_aoi_size_km"]),
                        )
                    st.rerun()

        st.divider()
        st.subheader("2. Choose AOI Dimension")
        size_presets = {"1 km": 1.0, "2 km (Standard)": 2.0, "5 km": 5.0, "10 km": 10.0, "Custom": None}
        current_size = float(st.session_state.get("cnn_aoi_size_km", 2.0))
        size_index = next((i for i, value in enumerate(size_presets.values()) if value is not None and abs(current_size - value) < 0.1), 4)
        size_col, custom_col = st.columns([3, 2])
        with size_col:
            size_choice = st.radio("AOI Extent Size Preset", list(size_presets), index=size_index,
                                   horizontal=True, key="cnn_aoi_size_choice")
        new_size = current_size
        with custom_col:
            if size_choice == "Custom":
                new_size = st.number_input("Custom Size (km)", min_value=0.5, max_value=50.0,
                                           value=current_size, step=0.5, key="cnn_custom_aoi_size")
            else:
                new_size = size_presets[size_choice]
        if abs(new_size - current_size) > 1e-4:
            st.session_state["cnn_aoi_size_km"] = new_size
            st.session_state["cnn_selected_bbox"] = bbox_from_center(
                float(st.session_state["cnn_center_lat"]), float(st.session_state["cnn_center_lon"]), new_size,
            )
            st.rerun()

        bbox = st.session_state["cnn_selected_bbox"]
        center_lat = (bbox.south + bbox.north) / 2.0
        center_lon = (bbox.west + bbox.east) / 2.0
        width_km = bbox.approx_width_m() / 1000.0
        height_km = bbox.approx_height_m() / 1000.0
        st.session_state["cnn_center_lat"] = center_lat
        st.session_state["cnn_center_lon"] = center_lon
        info_cols = st.columns([3, 2])
        with info_cols[0]:
            st.markdown(f"**Target Location:** {st.session_state['cnn_location_label']}")
            st.markdown(f"**Center Coordinates:** `{center_lat:.5f}\u00b0 N, {center_lon:.5f}\u00b0 E`")
            st.markdown(f"**Extent Dimension:** `~{width_km:.2f} km \u00d7 {height_km:.2f} km` (Source: *{st.session_state['cnn_aoi_source']}*)")
        with info_cols[1]:
            st.markdown(
                f"**Bounding Box (WGS84):**\n"
                f"- West: `{bbox.west:.5f}` | East: `{bbox.east:.5f}`\n"
                f"- South: `{bbox.south:.5f}` | North: `{bbox.north:.5f}`"
            )

        st.divider()
        st.subheader("3. Interactive Map")
        st.caption("Map view of the active acquisition area and center pin.")
        try:
            import folium
            from streamlit_folium import st_folium
            zoom = 15 if max(width_km, height_km) <= 1.5 else 14 if max(width_km, height_km) <= 3 else 13 if max(width_km, height_km) <= 7 else 12 if max(width_km, height_km) <= 15 else 11
            f_map = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)
            folium.Marker(location=[center_lat, center_lon], popup=st.session_state["cnn_location_label"],
                          tooltip="AOI Center", icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")).add_to(f_map)
            folium.Rectangle(bounds=[[bbox.south, bbox.west], [bbox.north, bbox.east]], color="#FF3333",
                             fill=True, fill_opacity=0.18, weight=2,
                             tooltip=f"Active AOI (~{width_km:.2f} km x {height_km:.2f} km)").add_to(f_map)
            st_folium(f_map, width=1100, height=390, key="cnn_aoi_folium_map")
        except Exception as exc:
            st.warning(f"Map component notice: {exc}")

        grid_px = st.select_slider("Acquisition Download Grid", options=[128, 256, 512], value=256,
                                   key="cnn_slider_raw_grid_px")
        st.caption("Raw acquisition dimension. Downstream spatial harmonization resamples to the authoritative 10m UTM grid.")
        valid, validation_error = validate_aoi(bbox)
        if valid:
            st.success(
                f"~{width_km:.2f} km \u00d7 {height_km:.2f} km | Download Grid: {grid_px}\u00d7{grid_px} px"
            )
        else:
            st.error(f"AOI validation error: {validation_error}")

    with st.expander("Step 2 - Date Range", expanded=True):
        date_cols = st.columns(2)
        with date_cols[0]:
            start_date = st.date_input("Start Date", value=datetime.date(2024, 9, 1), key="cnn_start_date")
        with date_cols[1]:
            end_date = st.date_input("End Date", value=datetime.date(2024, 9, 30), key="cnn_end_date")
        date_valid = start_date < end_date
        if not date_valid:
            st.error("Start date must be strictly before end date.")

    with st.expander("Step 3 - Search CDSE Catalogue", expanded=True):
        search_col, offline_col = st.columns(2)
        with search_col:
            search_clicked = st.button(
                "Search CDSE Catalogue (Live)", key="cnn_search_cdse", type="primary",
                use_container_width=True,
                disabled=not (valid and date_valid and validate_aoi(st.session_state["cnn_selected_bbox"])[0]),
            )
        with offline_col:
            offline_clicked = st.button("Load Cached Rome Scenes (Offline Demo)", key="cnn_offline_scenes", use_container_width=True)

        if search_clicked:
            for key in ("cnn_s1_scenes", "cnn_s2_scenes", "cnn_selected_s1_idx", "cnn_selected_s2_idx",
                        "cnn_s1_scene_widget", "cnn_s2_scene_widget", "cnn_auto_s1_for_s2_idx"):
                st.session_state.pop(key, None)
            try:
                from backend.services.satellite.cdse_sentinelhub import CDSESentinelHubProvider
                provider = CDSESentinelHubProvider()
                st.session_state["cnn_catalogue_provider"] = provider
                aoi = st.session_state["cnn_selected_bbox"]
                with st.spinner("Searching Sentinel-2 and Sentinel-1 on CDSE..."):
                    st.session_state["cnn_s2_scenes"] = provider.search_sentinel2(
                        bbox=aoi, start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"), max_results=25,
                    )
                    st.session_state["cnn_s1_scenes"] = provider.search_sentinel1(
                        bbox=aoi, start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"), max_results=25,
                    )
            except Exception as exc:
                st.error(f"CDSE catalogue search failed: {exc}")
                st.session_state["cnn_s2_scenes"] = []
                st.session_state["cnn_s1_scenes"] = []

        if offline_clicked:
            st.session_state.pop("cnn_s1_scene_widget", None)
            st.session_state.pop("cnn_s2_scene_widget", None)
            st.session_state.pop("cnn_auto_s1_for_s2_idx", None)
            st.session_state["cnn_selected_bbox"] = BoundingBox(west=12.4500, south=41.9000, east=12.4730, north=41.9230)
            st.session_state["cnn_location_label"] = "Rome Benchmark (Cached)"
            st.session_state["cnn_aoi_source"] = "Preset: Rome Benchmark (Cached)"
            st.session_state["cnn_center_lat"] = 41.9115
            st.session_state["cnn_center_lon"] = 12.4615
            st.session_state["cnn_aoi_size_km"] = 2.0
            st.session_state["cnn_s2_scenes"] = [S2Scene(
                scene_id="S2A_MSIL1C_20240921T100031_N0511_R122_T32TQM_20240921T120138.SAFE",
                dt=datetime.datetime(2024, 9, 21, 10, 0, 31, tzinfo=datetime.timezone.utc),
                cloud_cover=4.26, processing_level="L1C", footprint=None,
            )]
            st.session_state["cnn_s1_scenes"] = [S1Scene(
                scene_id="S1A_IW_GRDH_1SDV_20240920T051201_20240920T051226_055744_06CF2B_B9A8_COG.SAFE",
                dt=datetime.datetime(2024, 9, 20, 5, 12, 1, tzinfo=datetime.timezone.utc),
                product_type="GRD", polarizations=["VV", "VH"], orbit_direction="DESCENDING", instrument_mode="IW", footprint=None,
            )]
            st.session_state["cnn_selected_s2_idx"] = 0
            st.session_state["cnn_selected_s1_idx"] = 0
            st.rerun()

        s2_scenes = st.session_state.get("cnn_s2_scenes", [])
        s1_scenes = st.session_state.get("cnn_s1_scenes", [])
        if "cnn_s2_scenes" in st.session_state:
            st.subheader(f"Sentinel-2 L1C Results ({len(s2_scenes)} scenes)")
            if not s2_scenes:
                st.warning("No Sentinel-2 scenes found for this AOI and date range.")
            else:
                st.dataframe(pd.DataFrame([{
                    "#": index + 1, "Date": scene.utc_date(), "Time UTC": scene.utc_time(),
                    "Cloud %": scene.cloud_str().strip(), "Scene ID": scene.scene_id,
                } for index, scene in enumerate(s2_scenes)]), use_container_width=True, hide_index=True)
                s2_options = {
                    f"[{index + 1}] {scene.utc_date()} | Cloud: {scene.cloud_str().strip()} | {scene.scene_id[:45]}": index
                    for index, scene in enumerate(s2_scenes)
                }
                current = st.session_state.get("cnn_selected_s2_idx")
                choice = st.selectbox(
                    "Select Sentinel-2 scene", list(s2_options),
                    index=current if current is not None and current < len(s2_options) else None,
                    placeholder="-- Choose an S2 scene --", key="cnn_s2_scene_widget",
                )
                if choice is not None:
                    st.session_state["cnn_selected_s2_idx"] = s2_options[choice]
        selected_s2_idx = st.session_state.get("cnn_selected_s2_idx")
        st.session_state.pop("cnn_s1_scene_widget", None)  # Clear stale state from the old manual selector.
        auto_s1_idx = (
            best_s1_scene_index(s2_scenes[selected_s2_idx], s1_scenes)
            if selected_s2_idx is not None and selected_s2_idx < len(s2_scenes)
            else None
        )
        st.session_state["cnn_selected_s1_idx"] = auto_s1_idx
        if auto_s1_idx is not None:
            auto_scene = s1_scenes[auto_s1_idx]
            gap = abs((auto_scene.dt - s2_scenes[selected_s2_idx].dt).total_seconds()) / 86400.0
            st.info(f"Automatically selected closest Sentinel-1 GRD: {gap:.2f} days from S2 · {auto_scene.pol_str()}.")
        elif s1_scenes:
            st.warning("Select a Sentinel-2 scene to automatically match its closest Sentinel-1 GRD scene.")
        if "cnn_s1_scenes" in st.session_state:
            st.subheader(f"Sentinel-1 GRD Results ({len(s1_scenes)} scenes)")
            if not s1_scenes:
                st.warning("No Sentinel-1 scenes found for this AOI and date range.")
            else:
                st.dataframe(pd.DataFrame([{
                    "#": index + 1, "Date": scene.utc_date(), "Polarization": scene.pol_str(),
                    "Orbit": scene.orbit_direction or "?", "Mode": scene.instrument_mode or "?",
                    "VV+VH": "Yes" if scene.has_vv_vh() else "No", "Scene ID": scene.scene_id,
                } for index, scene in enumerate(s1_scenes)]), use_container_width=True, hide_index=True)
                if auto_s1_idx is None:
                    st.warning("No Sentinel-1 scene can be matched until a Sentinel-2 scene is selected.")
                else:
                    st.caption(f"Scene {auto_s1_idx + 1} is selected automatically based on the shortest time gap.")

    selected_pair = st.session_state.get("cnn_selected_bbox") and st.session_state.get("cnn_selected_s2_idx") is not None and st.session_state.get("cnn_selected_s1_idx") is not None and bool(st.session_state.get("cnn_s2_scenes")) and bool(st.session_state.get("cnn_s1_scenes"))
    cnn_s2_scene = st.session_state["cnn_s2_scenes"][st.session_state["cnn_selected_s2_idx"]] if selected_pair else None
    cnn_s1_scene = st.session_state["cnn_s1_scenes"][st.session_state["cnn_selected_s1_idx"]] if selected_pair else None
    temporal_gap_days = abs((cnn_s2_scene.dt - cnn_s1_scene.dt).total_seconds()) / 86400.0 if selected_pair else None
    temporal_pair_usable = selected_pair and temporal_gap_days <= 7.0
    with st.expander("Step 4 - Confirm Scene Selection & Temporal Alignment", expanded=True):
        confirm_cols = st.columns(2)
        with confirm_cols[0]:
            if cnn_s2_scene:
                st.success(
                    f"**Selected S2:** {cnn_s2_scene.utc_date()} {cnn_s2_scene.utc_time()} UTC | "
                    f"Cloud: {cnn_s2_scene.cloud_str().strip()} | {cnn_s2_scene.scene_id[:35]}..."
                )
            else:
                st.info("No Sentinel-2 scene selected.")
        with confirm_cols[1]:
            if cnn_s1_scene:
                st.success(
                    f"**Selected S1:** {cnn_s1_scene.utc_date()} {cnn_s1_scene.utc_time()} UTC | "
                    f"{cnn_s1_scene.pol_str()} {cnn_s1_scene.orbit_direction or '?'} | {cnn_s1_scene.scene_id[:35]}..."
                )
            else:
                st.info("No Sentinel-1 scene selected.")
        if selected_pair:
            delta_seconds = abs((cnn_s2_scene.dt - cnn_s1_scene.dt).total_seconds())
            delta_days = int(delta_seconds // 86400)
            delta_hours = int((delta_seconds % 86400) // 3600)
            gap_text = (
                f"S2: {cnn_s2_scene.dt.strftime('%Y-%m-%d %H:%M')} UTC | "
                f"S1: {cnn_s1_scene.dt.strftime('%Y-%m-%d %H:%M')} UTC | "
                f"Temporal gap: **{delta_days}d {delta_hours}h**"
            )
            if temporal_gap_days > 7:
                st.warning(f"{gap_text} | Warning: Temporal gap exceeds 7 days. CNN acquisition is disabled for this pair.")
            elif temporal_gap_days > 3:
                st.info(f"{gap_text} | Temporal gap is between 3 and 7 days. A closer pair is preferred.")
            else:
                st.success(f"{gap_text} | Excellent temporal alignment (within 3 days)!")
        else:
            st.caption("Select both an S2 scene and an S1 scene above to evaluate temporal gap and enable acquisition.")

    with st.expander("Step 5 - Acquire, Harmonize & Run DSen2-CR", expanded=True):
        cnn_service = service_getter()
        st.caption(f"Compute device: {cnn_service.device}")
        st.caption("CNN live acquisition requests Sentinel-1 Sigma0 dB to match the model's SEN12MS-CR training convention.")
        run_clicked = st.button("Acquire + Harmonize + Run DSen2-CR", key="cnn_live_acquire_run", type="primary", use_container_width=True, disabled=not temporal_pair_usable)
        if run_clicked:
            try:
                with st.spinner("Acquiring the selected CDSE scenes and harmonizing them on the common 10 m grid..."):
                    pair = acquire_and_harmonize()
                scene_date = cnn_s2_scene.utc_date()
                with st.spinner(f"Running DSen2-CR on {cnn_service.device}..."):
                    result = run_inference(
                        pair.s1_path, pair.s2_path, pair.validity_mask_path,
                        "cnn_live_result", "cnn_live_sr_result", "Live CDSE acquisition", brightness,
                        bbox=st.session_state["cnn_selected_bbox"], target_date=scene_date, harm_pair=pair,
                    )
                st.success("CDSE acquisition, harmonization, and DSen2-CR reconstruction completed.")
            except Exception as exc:
                st.error(f"CNN acquisition or reconstruction failed: {exc}")

    live_result = st.session_state.get("cnn_live_result")
    if live_result is not None:
        render_result(
            result=live_result,
            image_id=st.session_state.get("cnn_live_result_date") or "live",
            key_prefix="cnn_live_result",
            sr_state_key="cnn_live_sr_result",
            harm_pair=st.session_state.get("cnn_live_result_harm_pair"),
            bbox=st.session_state.get("cnn_live_result_bbox"),
            target_date=st.session_state.get("cnn_live_result_date"),
        )


def render_cnn_workspace(brightness, service_getter, acquire_and_harmonize, model_input_selector,
                         run_inference, render_result):
    """Match the Former Model's Live and Cached workspaces with DSen2-CR inference."""
    live_tab, cached_tab = st.tabs(["Live Acquisition", "Cached Phase 2 Baseline"])

    with live_tab:
        _render_scene_search(acquire_and_harmonize, service_getter, run_inference, render_result, brightness)

    with cached_tab:
        st.header("Pre-Validated Phase 2 Harmonized Data")
        st.caption("Run DSen2-CR on saved harmonized sessions or custom GeoTIFFs. For best compatibility, S1 should contain Sigma0 backscatter in dB (VV, VH) and S2 should contain 13 Sentinel-2 DN bands.")
        source_mode = st.radio(
            "Input Source",
            ["Cached Phase 2 CDSE Data", "Upload Custom Harmonized GeoTIFFs"],
            horizontal=True,
            key="cnn_cached_input_mode",
        )
        s1_path = s2_path = mask_path = None
        source_label = ""
        temporary = None
        if source_mode == "Cached Phase 2 CDSE Data":
            s1_path, s2_path, mask_path, source_label = model_input_selector("cnn_cached_pair_selector")
            if s1_path is not None:
                st.success(f"Input ready: {source_label}")
                with st.expander("Input file details", expanded=False):
                    st.code(f"S1: {s1_path}\nS2: {s2_path}\nMask: {mask_path or 'not supplied'}")
        else:
            upload_cols = st.columns(3)
            with upload_cols[0]:
                s1_upload = st.file_uploader("S1 SAR (2-band TIFF)", type=["tif", "tiff"], key="cnn_cached_s1_upload")
            with upload_cols[1]:
                s2_upload = st.file_uploader("S2 Optical (13-band TIFF)", type=["tif", "tiff"], key="cnn_cached_s2_upload")
            with upload_cols[2]:
                mask_upload = st.file_uploader("Validity Mask (optional)", type=["tif", "tiff"], key="cnn_cached_mask_upload")
            if s1_upload and s2_upload:
                temporary = tempfile.TemporaryDirectory(prefix="cnn_harmonized_upload_")
                temporary_dir = Path(temporary.name)
                s1_path, s2_path = temporary_dir / "s1_harmonized.tif", temporary_dir / "s2_harmonized.tif"
                s1_path.write_bytes(s1_upload.getvalue())
                s2_path.write_bytes(s2_upload.getvalue())
                if mask_upload:
                    mask_path = temporary_dir / "validity_mask.tif"
                    mask_path.write_bytes(mask_upload.getvalue())
                source_label = "Uploaded custom harmonized GeoTIFFs"

        if s1_path and s2_path:
            try:
                st.subheader("Input Previews")
                with rasterio.open(s1_path) as s1_src, rasterio.open(s2_path) as s2_src:
                    s1_preview = sar_preview(s1_src.read())
                    s2_preview = to_rgb(s2_src.read(), brightness=brightness)
                mask_preview = None
                if mask_path is not None and Path(mask_path).is_file():
                    with rasterio.open(mask_path) as mask_src:
                        mask_array = mask_src.read(1).astype(bool)
                    from PIL import Image
                    mask_preview = Image.fromarray(mask_array.astype("uint8") * 255, mode="L")
                preview_cols = st.columns(3)
                with preview_cols[0]:
                    st.image(s1_preview, caption="S1 SAR (VV/VH False Color)", use_container_width=True)
                with preview_cols[1]:
                    st.image(s2_preview, caption="S2 Cloudy Optical (RGB)", use_container_width=True)
                with preview_cols[2]:
                    if mask_preview is not None:
                        st.image(mask_preview, caption=f"Spatial Validity Mask ({int(mask_array.sum()):,} valid px)", use_container_width=True)
                    else:
                        st.caption("No spatial validity mask supplied.")
            except Exception as exc:
                st.warning(f"Input preview could not be rendered: {exc}")
            if st.button("Run DSen2-CR Cloud Removal", type="primary", use_container_width=True, key="cnn_cached_run"):
                try:
                    with st.spinner(f"Running DSen2-CR on {service_getter().device}..."):
                        run_inference(
                            s1_path, s2_path, mask_path,
                            "cnn_cached_result", "cnn_cached_sr_result", source_label, brightness,
                        )
                except Exception as exc:
                    st.error(f"DSen2-CR inference failed: {exc}")
            if temporary is not None:
                temporary.cleanup()

        cached_result = st.session_state.get("cnn_cached_result")
        if cached_result is not None:
            render_result(
                result=cached_result,
                image_id="cached_phase2",
                key_prefix="cnn_cached_result",
                sr_state_key="cnn_cached_sr_result",
            )
