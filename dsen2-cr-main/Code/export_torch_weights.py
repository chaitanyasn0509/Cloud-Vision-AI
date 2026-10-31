"""Export the Keras HDF5 convolution weights to a NumPy archive for PyTorch."""
from __future__ import print_function

import sys

import h5py
import numpy as np


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: export_torch_weights.py MODEL.hdf5 OUTPUT.npz")
    source_path, output_path = sys.argv[1:]
    arrays = {}
    with h5py.File(source_path, "r") as source:
        for layer_index in range(1, 35):
            name = "conv2d_{}".format(layer_index)
            group = source[name][name + "_1"]
            arrays[name + ".kernel"] = np.asarray(group["kernel:0"], dtype=np.float32)
            arrays[name + ".bias"] = np.asarray(group["bias:0"], dtype=np.float32)
    np.savez(output_path, **arrays)
    print("Exported 34 convolution layers to {}".format(output_path))


if __name__ == "__main__":
    main()
