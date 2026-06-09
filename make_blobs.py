#!/usr/bin/env python3
"""
make_blobs.py

Converts OpenVINO IR (.xml/.bin) files to DepthAI .blob files.
Run from your depthai_hand_tracker directory:
    python make_blobs.py
"""

import blobconverter
import os

os.makedirs("models/blobs", exist_ok=True)

# ---------------------------------------------------------------------------
# Each entry: (display_name, xml_path, bin_path, extra_compile_params)
#
# hand_detector:
#   - Fixed path separator (was missing backslash causing name mangling)
#   - No special params needed for palm detection
#
# canned_gesture_classifier:
#   - Has dynamic input shape — must freeze it with -freeze_model_output
#     and specify static shape via compile params so the Myriad VPU
#     can allocate fixed memory buffers.
#   - Input 'hand_embedding' is shape [1, 128] per the model card.
# ---------------------------------------------------------------------------

models = [
    (
        "hand_detector",
        "models/ir/hand_detector/hand_detector.xml",
        "models/ir/hand_detector/hand_detector.bin",
        [],  # no extra params needed
    ),
    (
        "canned_gesture_classifier",
        "models/ir/canned_gesture_classifier/canned_gesture_classifier.xml",
        "models/ir/canned_gesture_classifier/canned_gesture_classifier.bin",
        ["-iop", "hand_embedding:FP16"],  # freeze dynamic input to known type
    ),
]

# These two already compiled successfully — skip them.
# Remove from this list if you want to recompile them.
already_done = {
    "hand_landmarks_detector",
    "gesture_embedder",
}

for name, xml, bin_, compile_params in models:
    if name in already_done:
        print(f"Skipping {name} (already compiled)")
        continue

    # Verify IR files exist before attempting upload
    if not os.path.exists(xml):
        print(f"FAILED: {name} — XML not found at: {xml}")
        continue
    if not os.path.exists(bin_):
        print(f"FAILED: {name} — BIN not found at: {bin_}")
        continue

    print(f"Compiling {name}...")
    try:
        blob_path = blobconverter.from_openvino(
            xml=xml,
            bin=bin_,
            shaves=4,
            output_dir="models/blobs",
            version=2023.1,
            compile_params=compile_params if compile_params else None,
        )
        print(f"  OK -> {blob_path}")
    except Exception as e:
        print(f"  FAILED: {e}")

print("\nDone.")
print("\nBlobs in models/blobs/:")
for f in os.listdir("models/blobs"):
    path = os.path.join("models/blobs", f)
    print(f"  {f}  ({os.path.getsize(path):,} bytes)")
