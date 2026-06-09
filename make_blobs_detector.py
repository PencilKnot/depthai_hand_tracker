#!/usr/bin/env python3
"""
make_blobs.py — fixes opset11 Interpolate error by trying newer OpenVINO versions.
Run from your depthai_hand_tracker directory: python make_blobs.py
"""

import blobconverter
import os

os.makedirs("models/blobs", exist_ok=True)

XML = "models/ir/hand_detector/hand_detector.xml"
BIN = "models/ir/hand_detector/hand_detector.bin"

if not os.path.exists(XML):
    print(f"ERROR: XML not found at {XML}")
    raise SystemExit(1)

# Try progressively newer OpenVINO versions until one works.
# The opset11 Interpolate op is supported from 2022.3 onward.
VERSIONS_TO_TRY = ["2022.3", "2023.1", "2023.2"]

success = False
for version in VERSIONS_TO_TRY:
    print(f"Trying OpenVINO version {version}...")
    try:
        blob_path = blobconverter.from_openvino(
            xml=XML,
            bin=BIN,
            shaves=4,
            version=version,
            output_dir="models/blobs",
        )
        print(f"  SUCCESS with {version} -> {blob_path}")
        success = True
        break
    except Exception as e:
        err = str(e)
        if "opset" in err.lower() or "unsupported" in err.lower():
            print(f"  opset error with {version}, trying next...")
        elif "404" in err or "not found" in err.lower():
            print(f"  version {version} not available on server, trying next...")
        else:
            print(f"  FAILED with {version}: {e}")

if not success:
    print("""
All versions failed. The model uses ops too new for blobconverter's server.
Fall back to Option 2: re-convert the tflite with an older opset.

Run this command to reconvert with opset10 compatibility:

    mo --input_model "models/gesture_recognizer_extracted/hand_landmarker_extracted/hand_detector.tflite" ^
       --framework tflite ^
       --output_dir "models/ir/hand_detector_opset10" ^
       --model_name hand_detector ^
       --compress_to_fp16 True

Then update XML/BIN paths in this script to point to hand_detector_opset10
and run again with version="2022.1".
""")
