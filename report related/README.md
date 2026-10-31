# Model documentation

This folder documents the two cloud-removal paths available in the CloudVisionAI application.

- [Former Model (ECRformer)](Former_Model_README.md) — input handling, tiled inference, evaluation, and exports.
- [CNN Model (DSen2-CR)](CNN_Model_README.md) — SAR/optical inputs, inference, cloud-guided output, evaluation, and exports.

Both models expect spatially aligned Sentinel-1 and Sentinel-2 inputs. The Comparison tab runs both models sequentially on one selected pair to support a controlled comparison. The optional OpenSR stage is post-processing and produces a separate four-band RGB+NIR product.

The Former and CNN tabs keep independent widget and session-state namespaces. Running the CNN tab does not replace the Former tab's selected inputs or results. The Comparison tab intentionally uses one shared pair and runs Former followed by CNN.
