# Former Model — ECRformer

## Role in the application

Former Model is the application's ECRformer cloud-removal workflow. It is the original model path in the app and is implemented by `InferenceService`, using the project checkpoint `model.ckpt`. The application can run it on the GPU when CUDA is available, or use the device selected by the runtime.

## Inputs and data preparation

The model consumes a spatially matched pair:

| Input | Expected content |
| --- | --- |
| Sentinel-1 | Two bands ordered VV, VH; calibrated backscatter in dB |
| Cloudy Sentinel-2 | Thirteen bands in project order: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12; normally DN in the 0–10,000 range |
| Validity mask | Optional one-band mask, where nonzero means a valid pixel |

The input rasters must share dimensions, CRS, and affine transform. Live CDSE acquisition searches for S1 GRD and S2 scenes, selects an S1 scene near the selected S2 acquisition, and uses the shared harmonization pipeline to reproject/co-register the data onto a common 10 m grid. Cached harmonized pairs and uploaded harmonized GeoTIFFs are also supported.

Before inference, Sentinel-2 values are clipped to 0–10,000 and scaled to reflectance [0, 1]. Sentinel-1 dB values are clipped to -25–0 dB and linearly scaled to [0, 1]. Invalid pixels are set to zero and excluded from the valid-pixel mask.

## Inference workflow

1. The service validates band counts, raster dimensions, CRS, transforms, and an optional mask.
2. It concatenates the normalized S1 and S2 inputs into a 15-channel model input (2 SAR + 13 optical bands).
3. ECRformer processes the image in 128 × 128 tiles by default, with configurable overlap (32 pixels by default). Overlapping predictions are averaged into one output.
4. The 13 output bands are clipped to [0, 1]. Invalid pixels are assigned the configured NoData value.

The result object contains the prediction, valid mask, geospatial profile, previews, and diagnostics such as device, runtime, tile count, valid pixels, and per-band statistics.

## Evaluation and exports

When a clear reference is supplied, the application can calculate reference-based metrics including MAE, RMSE, PSNR, SSIM, spectral angle (SAM), and pixel count. Metrics are meaningful only when the reference has the same band order, scale, grid, and sufficiently similar acquisition conditions; masking and temporal differences should be documented in research reporting.

The authoritative cloud-removal export is a georeferenced 13-band GeoTIFF. The UI also offers an RGB GeoTIFF, PNG preview, diagnostics JSON, optional clear-reference evaluation, and optional OpenSR processing.

## Optional super-resolution

OpenSR is a separate post-processing stage; it does not change the 13-band ECRformer product. It selects Sentinel-2 B04, B03, B02, and B08 and generates a 4-band RGB+NIR GeoTIFF at nominal 2.5 m resolution (4× per side from 10 m input). A 10 m ground-truth reference can only score this output after it is area-averaged back to the reference grid. Those are 10 m-equivalent agreement metrics, not native 2.5 m detail-accuracy measurements.

## Main implementation files

- `backend/services/processing/inference_service.py` — validation, normalization, tiling, inference, and output container.
- `models/ecrformer_model.py` and related modules in `models/` — ECRformer network definition.
- `preprocessing.py` — mask-safe S1/S2 normalization.
- `backend/services/processing/harmonizer.py` — common-grid harmonization.
- `app.py` — Former Model interface, result rendering, evaluation, exports, and OpenSR controls.
