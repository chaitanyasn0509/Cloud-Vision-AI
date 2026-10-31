# CNN Model — DSen2-CR

## Role in the application

CNN Model runs the bundled DSen2-CR CARL network through the project's PyTorch inference adapter, `DSen2CRService`. It is a pretrained reconstruction model; this application performs inference and does not train or fine-tune it. CUDA is used when available. The PyTorch-compatible converted checkpoint is stored at `dsen2-cr-main/checkpoints/model_SARcarl.torch.npz` after conversion from the bundled HDF5 model weights.

## Inputs and preparation

CNN expects a co-registered pair on the same grid:

| Input | Expected content |
| --- | --- |
| Sentinel-1 | Two bands ordered VV, VH, GRD backscatter in dB |
| Cloudy Sentinel-2 | Thirteen bands in project order (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12), generally DN in 0–10,000 |
| Validity mask | Optional one-band spatial validity mask |

CNN offers the same kinds of input workflow as Former Model: live CDSE acquisition and harmonization, cached harmonized sessions, or uploaded harmonized GeoTIFFs. Live selection prefers an S1 scene containing both VV and VH and then the smallest absolute time gap to the selected S2 scene. The selected scene can still be changed manually. S1, S2, and mask must match dimensions and CRS; S2 and S1 affine grids must match.

The model clips Sentinel-2 DN to 0–10,000, divides by 2,000 for its model-space optical input, and normalizes Sentinel-1 VV/VH from the trained dB ranges (VV -25–0 dB; VH -32.5–0 dB) to [0, 1]. Values outside those expected ranges are clipped. The interface exposes input-compatibility diagnostics; these describe range compatibility and are not accuracy measurements.

## Inference workflow

1. `DSen2CRService` validates the 13-band S2 and 2-band S1 pair and creates or reads a valid-pixel mask.
2. A spectral heuristic estimates where the cloudy input may need reconstruction. It is derived from optical spectral relationships; it is not a trained cloud-probability output or a manually supplied cloud mask.
3. The DSen2-CR network processes 128 × 128 tiles with 32-pixel overlap by default. Predictions are overlap-averaged, converted to reflectance, and clipped to [0, 1].
4. The result UI offers two product views: **Raw DSen2-CR**, which uses the network output across valid pixels, and **Cloud-guided blend**, which blends the model prediction into areas selected by the heuristic and retains more original cloudy-S2 detail elsewhere. The blend threshold is adjustable.

The cloud-guided view is designed to reduce unnecessary changes to clear areas. Its blend map is still a heuristic. It can miss thin clouds, haze, snow, bright roofs, or other spectrally similar surfaces; performance depends strongly on correct calibration, band order, co-registration, and the temporal relationship between S1 and S2. Inspect raw and blended outputs and report which output mode and threshold were evaluated.

## Results, evaluation, and exports

CNN results include a 13-band cloud-free GeoTIFF, RGB GeoTIFF, PNG preview, and diagnostics JSON. Where a reference is available, the UI evaluates the selected output against it using reference-based metrics such as MAE, RMSE, PSNR, SSIM, SAM, and valid-pixel count. For a fair model comparison, both outputs must use the same target image, band subset/order, scale, grid, and valid mask. A temporally different clear image is an imperfect reference and may penalize real land-cover change.

Comparison runs Former first and CNN second on the same input pair, then displays the outputs together. It can also run OpenSR separately on each cloud-removal result. OpenSR creates a four-band B04/B03/B02/B08 output; it is post-processing, not part of DSen2-CR. When evaluation uses a 10 m reference, the 2.5 m output is area-averaged to that reference grid, so the score does not measure native 2.5 m detail accuracy.

## Main implementation files

- `backend/services/processing/dsen2cr_service.py` — PyTorch network adapter, input scaling, tiling, spectral cloud score, and inference result.
- `dsen2-cr-main/Code/export_torch_weights.py` — conversion of the original HDF5 weights to the PyTorch-compatible NPZ checkpoint.
- `ui/cnn_workspace.py` — CNN live acquisition, cached/uploaded inputs, and CNN tab controls.
- `app.py` — shared acquisition/harmonization, cloud-guided result rendering, evaluation, Comparison, exports, and OpenSR integration.
