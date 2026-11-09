"""Sanity check script for MRI, US, and CT baseline DenseNet121 models.

Reads test images from demo_samples/baseline/<model>/<true_class>/
Example: demo_samples/baseline/mri/glioma/img1.jpg
Prints predicted label, top probability, and match status, then reports overall accuracy.
If demo_samples/baseline does not exist, prints instructions on how to create it and exits.
Does NOT run against raw demo_samples/ or invent results.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root and AI/ to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
AI_DIR = REPO_ROOT / "AI"
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

# Ensure flag is enabled for evaluation
os.environ.setdefault("ENABLE_BASELINE_MODELS", "1")

from predict import predict_baseline
from preprocessing import pad_square
import torch
from torchvision import transforms as T
from PIL import Image


def main():
    base_dir = REPO_ROOT / "demo_samples" / "baseline"
    if not base_dir.exists() or not base_dir.is_dir():
        print(f"[!] Baseline samples directory not found: {base_dir}")
        print("\nTo run sanity checks, organize test images in the following structure:")
        print("  demo_samples/baseline/<model>/<true_class>/<image_file>")
        print("\nExamples:")
        print("  demo_samples/baseline/mri/glioma/scan1.jpg")
        print("  demo_samples/baseline/mri/meningioma/scan2.jpg")
        print("  demo_samples/baseline/mri/notumor/scan3.jpg")
        print("  demo_samples/baseline/mri/pituitary/scan4.jpg")
        print("  demo_samples/baseline/us/benign/us1.png")
        print("  demo_samples/baseline/us/malignant/us2.png")
        print("  demo_samples/baseline/us/normal/us3.png")
        print("  demo_samples/baseline/ct/adenocarcinoma/ct1.png")
        print("  demo_samples/baseline/ct/large_cell/ct2.png")
        print("  demo_samples/baseline/ct/normal/ct3.png")
        print("  demo_samples/baseline/ct/squamous/ct4.png")
        print("\nPlease create the directory structure and supply test images.")
        sys.exit(0)

    models = [d.name.lower() for d in base_dir.iterdir() if d.is_dir()]
    if not models:
        print(f"[!] No model subdirectories found in {base_dir}")
        sys.exit(0)

    overall_results = {}

    for model_name in sorted(models):
        if model_name not in {"mri", "us", "ct"}:
            print(f"Skipping unknown model folder: {model_name}")
            continue

        model_dir = base_dir / model_name
        class_dirs = [d for d in model_dir.iterdir() if d.is_dir()]

        if not class_dirs:
            print(f"[!] No class subdirectories in {model_dir}")
            continue

        print(f"\n==========================================")
        print(f"Evaluating Baseline Model: {model_name.upper()}")
        print(f"==========================================")

        total = 0
        correct = 0
        padded_correct = 0

        for class_dir in sorted(class_dirs):
            true_class = class_dir.name
            image_files = [
                f for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".dcm"}
            ]

            print(f"\nClass '{true_class}' ({len(image_files)} images):")
            for img_path in sorted(image_files):
                try:
                    res = predict_baseline(model_name, str(img_path))
                    pred_label = res["primary_label"]
                    conf = res["confidence"]
                    is_match = (pred_label.lower() == true_class.lower())

                    total += 1
                    if is_match:
                        correct += 1

                    match_str = "MATCH" if is_match else f"MISMATCH (expected {true_class})"
                    print(f"  [{img_path.name}] -> {pred_label} ({conf * 100:.1f}%) | {match_str}")

                    # For MRI: also check padded variant
                    if model_name == "mri":
                        from dl_models import load_baseline
                        m, cls_list, temp, _ = load_baseline("mri")
                        raw_im = Image.open(img_path).convert("RGB")
                        pad_im = pad_square(raw_im)
                        tf = T.Compose([
                            T.Resize((224, 224)),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
                        t_pad = tf(pad_im).unsqueeze(0)
                        with torch.no_grad():
                            l_pad = m(t_pad)
                            p_pad = torch.softmax(l_pad / temp, dim=-1).squeeze(0)
                            pad_pred = cls_list[p_pad.argmax().item()]
                            if pad_pred.lower() == true_class.lower():
                                padded_correct += 1
                except Exception as exc:
                    print(f"  [{img_path.name}] -> ERROR: {exc}")

        acc = (correct / total) if total > 0 else 0.0
        print(f"\n--- {model_name.upper()} Summary ---")
        print(f"Total evaluated: {total}")
        print(f"Correct predictions: {correct}")
        print(f"Accuracy: {acc:.3f} ({acc * 100:.1f}%)")
        if model_name == "mri" and total > 0:
            padded_acc = padded_correct / total
            print(f"Padded pipeline accuracy: {padded_acc:.3f} ({padded_acc * 100:.1f}%)")
            if acc < padded_acc - 0.05:
                print("WARNING: Unpadded accuracy is notably lower than padded accuracy!")

        overall_results[model_name] = {"total": total, "correct": correct, "accuracy": acc}


if __name__ == "__main__":
    main()

