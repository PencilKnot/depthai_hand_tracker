#!/usr/bin/env python3
"""
extract_task_models.py

Extracts the individual .tflite model files from a MediaPipe .task bundle,
then guides you through converting them to DepthAI .blob files.

The .task file is a zip archive with alignment padding — standard unzip
misreads it, but Python's zipfile module handles it correctly.

Usage:
    python extract_task_models.py hand_landmarker.task
    python extract_task_models.py gesture_recognizer.task

What you get:
    hand_landmarker.task     → hand_detector.tflite
                               hand_landmarker.tflite

    gesture_recognizer.task  → hand_detector.tflite
                               hand_landmarker.tflite
                               hand_gesture_classifier.tflite
                               (possibly more sub-bundle files)
"""

import zipfile
import os
import sys
import json


def extract_task(task_path: str, output_dir: str = None):
    if not os.path.exists(task_path):
        print(f"ERROR: File not found: {task_path}")
        sys.exit(1)

    if output_dir is None:
        base = os.path.splitext(os.path.basename(task_path))[0]
        output_dir = os.path.join(os.path.dirname(task_path), base + "_extracted")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nReading: {task_path}  ({os.path.getsize(task_path):,} bytes)")
    print(f"Output:  {output_dir}\n")

    if not zipfile.is_zipfile(task_path):
        print("ERROR: This file is not a valid zip/task bundle.")
        print("Make sure you downloaded the file completely and it is not corrupted.")
        sys.exit(1)

    extracted = []
    with zipfile.ZipFile(task_path, "r") as zf:
        names = zf.namelist()
        print(f"Files inside bundle ({len(names)} total):")
        for name in names:
            info = zf.getinfo(name)
            print(f"  {name:50s}  {info.file_size:>10,} bytes")

        print()
        for name in names:
            dest = os.path.join(output_dir, name)
            # Handle nested paths inside the zip
            os.makedirs(
                os.path.dirname(dest) if os.path.dirname(dest) else output_dir,
                exist_ok=True,
            )
            data = zf.read(name)
            with open(dest, "wb") as f:
                f.write(data)
            extracted.append((name, dest, len(data)))
            print(f"  Extracted: {name}  →  {dest}")

    print(f"\nExtracted {len(extracted)} file(s) to {output_dir}")

    # Check for nested .task bundles (gesture_recognizer bundles hand_landmarker inside)
    sub_tasks = [d for n, d, _ in extracted if d.endswith(".task")]
    if sub_tasks:
        print("\nFound nested .task bundles — extracting those too:")
        for sub in sub_tasks:
            sub_out = os.path.splitext(sub)[0] + "_extracted"
            extract_task(sub, sub_out)

    # Summarise what we found
    all_tflite = []
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".tflite"):
                all_tflite.append(os.path.join(root, f))

    print("\n" + "=" * 60)
    print("TFLITE FILES FOUND:")
    print("=" * 60)
    # make sure that all the tflite fils are accounted for
    if not all_tflite:
        print("  None found. The bundle may use a different internal format.")
        print("  Try inspecting the files in the output directory manually.")
        return

    for p in all_tflite:
        print(f"  {p}  ({os.path.getsize(p):,} bytes)")

    print()
    print_conversion_instructions(all_tflite)


def print_conversion_instructions(tflite_files: list):
    print("=" * 60)
    print("NEXT STEPS: CONVERT TO BLOB")
    print("=" * 60)
    print("""
The .tflite files above are what you convert to .blob for the OAK-D.
Run these commands in order:

--- Step 1: Install conversion tools ---

    pip install openvino-dev blobconverter

--- Step 2: Convert each .tflite to OpenVINO IR (.xml + .bin) ---
""")
    for tflite in tflite_files:
        name = os.path.splitext(os.path.basename(tflite))[0]
        print(f"    mo \\")
        print(f'      --input_model "{tflite}" \\')
        print(f"      --output_dir ./ir/{name} \\")
        print(f"      --model_name {name}")
        print()

    print("""--- Step 3: Read the layer names from the .xml files ---

    The .xml files are human-readable. Open them and search for
    <layer> tags — the "name" attribute is the layer string you use
    in HandTracker.py's getLayerFp16("...") calls.

    Specifically look for the OUTPUT layers near the bottom of the xml.
    These are the names you need to update in HandTracker.py.

--- Step 4: Compile IR to .blob ---
""")

    print('    python -c "')
    print("    import blobconverter")
    for tflite in tflite_files:
        name = os.path.splitext(os.path.basename(tflite))[0]
        varname = name.replace("-", "_")
        print(f"""
    {varname}_blob = blobconverter.from_openvino(
        xml='./ir/{name}/{name}.xml',
        bin='./ir/{name}/{name}.bin',
        shaves=4,
        output_dir='./blobs'
    )
    print('{name} blob:', {varname}_blob)""")
    print('    "')

    print("""
--- Step 5: Update HandTracker.py ---

    After conversion, update these values in HandTracker.py:

    1. self.pd_input_length = 192   (line ~209, change from 128)

    2. In pd_postprocess(), update anchor count if needed:
       The newer palm detection model uses 2016 anchors (not 896).
       Check the xml output shape to confirm.

    3. In lm_postprocess(), update getLayerFp16() string names
       to match what you read from the landmark model's .xml file.

    4. Point pd_model and lm_model args to your new .blob paths:
       python demo.py --pd_model blobs/hand_detector.blob \\
                      --lm_model blobs/hand_landmarker.blob

--- Checking input dimensions from the .xml ---

    In the xml, find the <layer type="Input"> element.
    The <dim> values tell you the input shape, e.g.:
      <dim>1</dim>  <- batch
      <dim>3</dim>  <- channels (RGB)
      <dim>192</dim> <- height  ← this tells you pd_input_length
      <dim>192</dim> <- width

--- Checking anchor count ---

    In the xml for palm detection, find the output layer.
    Its shape will be [1, N, 18] where N = number of anchors.
    N=896  → 128px model  (old)
    N=2016 → 192px model  (current)
    Update nb_anchors and generate_handtracker_anchors() accordingly.
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_task_models.py <path_to_file.task>")
        print()
        print("Example:")
        print("  python extract_task_models.py models/hand_landmarker.task")
        print("  python extract_task_models.py models/gesture_recognizer.task")
        sys.exit(0)

    extract_task(sys.argv[1])
