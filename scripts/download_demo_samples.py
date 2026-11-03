"""Generate high-fidelity anatomical radiology demo samples across modalities."""

from pathlib import Path
import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "demo_samples"


def generate_normal_cxr(h: int = 512, w: int = 512) -> np.ndarray:
    """Anatomically realistic normal PA chest radiograph with clear lung fields."""
    Y, X = np.ogrid[:h, :w]
    img = np.full((h, w), 180.0, dtype=np.float32)
    body = ((X - 256) / 235.0) ** 2 + ((Y - 260) / 245.0) ** 2 < 1.0
    img[~body] = 18.0

    spine = np.exp(-((X - 256.0) ** 2) / (2 * 16.0 ** 2)) * 40.0
    img += spine

    right_dome = 370.0 - 55.0 * np.sin(np.clip((X - 110.0) / 140.0 * np.pi, 0, np.pi))
    left_dome = 380.0 - 48.0 * np.sin(np.clip((X - 265.0) / 140.0 * np.pi, 0, np.pi))
    sub_diaphragm = ((Y > right_dome) & (X < 256)) | ((Y > left_dome) & (X >= 256))
    img[sub_diaphragm] = 200.0

    rlung = (((X - 165.0) / 78.0) ** 2 + ((Y - 225.0) / 125.0) ** 2 < 1.0) & (Y < right_dome) & (X > 50) & (X < 242) & (Y > 75)
    llung = (((X - 345.0) / 78.0) ** 2 + ((Y - 230.0) / 120.0) ** 2 < 1.0) & (Y < left_dome) & (X > 270) & (X < 460) & (Y > 80)
    img[rlung] -= 90.0
    img[llung] -= 90.0

    heart = (((X - 245.0) / 68.0) ** 2 + ((Y - 325.0) / 58.0) ** 2 < 1.0) & (X > 195) & (X < 335) & (Y > 265) & (Y < 385)
    img[heart] = 170.0

    for i in range(7):
        yc = 105.0 + i * 38.0
        rib_r = np.abs(Y - (yc + 0.0012 * (X - 165.0) ** 2)) < 5.0
        img[rib_r & rlung] += 22.0
        rib_l = np.abs(Y - (yc + 0.0012 * (X - 345.0) ** 2)) < 5.0
        img[rib_l & llung] += 22.0

    clav_r = (np.abs(Y - (95.0 + 0.1 * (X - 165.0))) < 6.0) & (X > 85) & (X < 245)
    clav_l = (np.abs(Y - (95.0 - 0.1 * (X - 345.0))) < 6.0) & (X > 265) & (X < 425)
    img[clav_r | clav_l] += 32.0

    np.random.seed(42)
    img += np.random.normal(0, 4.0, (h, w)).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0)


def generate_pneumonia_cxr(h: int = 512, w: int = 512) -> np.ndarray:
    """Anatomically realistic CXR with dense right lower lobe consolidation."""
    base = generate_normal_cxr(h, w).astype(np.float32)
    Y, X = np.ogrid[:h, :w]
    rll = (((X - 160.0) / 45.0) ** 2 + ((Y - 330.0) / 40.0) ** 2 < 1.0)
    base[rll] += 65.0
    infiltrate = np.random.normal(25.0, 10.0, (h, w)).astype(np.float32)
    base[rll] += infiltrate[rll]
    base = np.clip(base, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(base, (3, 3), 0)


def generate_ct_nodule(h: int = 512, w: int = 512) -> np.ndarray:
    """Axial CT chest slice with ribs, spine, low-density lungs, and a 14 mm nodule."""
    Y, X = np.ogrid[:h, :w]
    ct = np.full((h, w), 15.0, dtype=np.float32)

    chest_wall = ((X - 256) / 210.0) ** 2 + ((Y - 256) / 180.0) ** 2 < 1.0
    ct[chest_wall] = 160.0
    fat = ((X - 256) / 198.0) ** 2 + ((Y - 256) / 168.0) ** 2 < 1.0
    ct[fat] = 70.0
    muscle = ((X - 256) / 190.0) ** 2 + ((Y - 256) / 160.0) ** 2 < 1.0
    ct[muscle] = 160.0

    spine_bone = ((X - 256) / 28.0) ** 2 + ((Y - 380) / 24.0) ** 2 < 1.0
    ct[spine_bone] = 240.0
    canal = ((X - 256) / 10.0) ** 2 + ((Y - 375) / 10.0) ** 2 < 1.0
    ct[canal] = 40.0
    sternum = ((X - 256) / 22.0) ** 2 + ((Y - 105) / 10.0) ** 2 < 1.0
    ct[sternum] = 230.0

    for a in np.linspace(0.2, np.pi - 0.2, 6):
        rx, ry = int(256 + 185 * np.cos(a)), int(256 + 155 * np.sin(a))
        cv2.circle(ct, (rx, ry), 7, 235.0, -1)
        lx, ly = int(256 - 185 * np.cos(a)), int(256 + 155 * np.sin(a))
        cv2.circle(ct, (lx, ly), 7, 235.0, -1)

    ct_rlung = (((X - 175.0) / 65.0) ** 2 + ((Y - 250.0) / 115.0) ** 2 < 1.0) & (X > 90) & (X < 240)
    ct_llung = (((X - 337.0) / 65.0) ** 2 + ((Y - 250.0) / 115.0) ** 2 < 1.0) & (X > 272) & (X < 422)
    ct[ct_rlung | ct_llung] = 35.0

    mediastinum = ((X - 256) / 45.0) ** 2 + ((Y - 250) / 120.0) ** 2 < 1.0
    ct[mediastinum] = 145.0
    cv2.circle(ct, (240, 290), 16, 170.0, -1)

    # 14 mm solid pulmonary nodule in right lung field
    cv2.circle(ct, (160, 270), 12, 190.0, -1)
    for pt in [(185, 230), (170, 210), (195, 270), (325, 230), (335, 270)]:
        cv2.circle(ct, pt, 3, 110.0, -1)

    ct += np.random.normal(0, 3.0, (h, w)).astype(np.float32)
    ct = np.clip(ct, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(ct, (3, 3), 0)


def generate_mri_tumor(h: int = 512, w: int = 512) -> np.ndarray:
    """Axial T2/FLAIR brain MRI slice with skull, ventricles, and frontoparietal mass."""
    Y, X = np.ogrid[:h, :w]
    mri = np.full((h, w), 10.0, dtype=np.float32)

    scalp = ((X - 256) / 180.0) ** 2 + ((Y - 256) / 215.0) ** 2 < 1.0
    mri[scalp] = 120.0
    skull_bone = ((X - 256) / 170.0) ** 2 + ((Y - 256) / 205.0) ** 2 < 1.0
    mri[skull_bone] = 30.0

    brain = ((X - 256) / 162.0) ** 2 + ((Y - 256) / 197.0) ** 2 < 1.0
    mri[brain] = 135.0

    fissure = (np.abs(X - 256) < 2) & brain
    mri[fissure] = 40.0

    vent_r = (((X - 240) / 12.0) ** 2 + ((Y - 245) / 45.0) ** 2 < 1.0) & (X > 228) & (X < 252)
    vent_l = (((X - 272) / 12.0) ** 2 + ((Y - 245) / 45.0) ** 2 < 1.0) & (X > 260) & (X < 284)
    mri[vent_r | vent_l] = 40.0

    for r in range(40, 150, 25):
        for th in np.linspace(0, 2 * np.pi, 16):
            sx = int(256 + r * np.cos(th))
            sy = int(256 + (r * 1.2) * np.sin(th))
            if brain[sy, sx]:
                cv2.circle(mri, (sx, sy), 3, 45.0, -1)

    # Enhancing mass with perilesional edema in right frontal lobe
    edema = (((X - 195) / 36.0) ** 2 + ((Y - 180) / 32.0) ** 2 < 1.0)
    mri[edema & brain] = 175.0
    tumor_core = (((X - 195) / 20.0) ** 2 + ((Y - 180) / 18.0) ** 2 < 1.0)
    mri[tumor_core & brain] = 225.0

    mri += np.random.normal(0, 4.0, (h, w)).astype(np.float32)
    mri = np.clip(mri, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(mri, (3, 3), 0)


def generate_us_breast(h: int = 512, w: int = 512) -> np.ndarray:
    """High-frequency B-mode breast ultrasound with irregular hypoechoic mass (BI-RADS 4)."""
    Y, X = np.ogrid[:h, :w]
    us = np.full((h, w), 25.0, dtype=np.float32)

    us[40:55, :] = 195.0   # skin line
    us[55:120, :] = 70.0   # subcutaneous fat
    us[120:340, :] = 175.0 # fibroglandular parenchyma
    us[340:420, :] = 80.0  # retromammary fat / muscle

    speckle = np.random.normal(0, 22.0, (h, w)).astype(np.float32)
    us += speckle

    # Focal hypoechoic lesion (irregular, taller-than-wide, angular margins)
    lesion = (((X - 260) / 32.0) ** 2 + ((Y - 220) / 42.0) ** 2 < 1.0)
    us[lesion] = 42.0 + np.random.normal(0, 8.0, np.sum(lesion))

    # Posterior acoustic attenuation / shadowing
    shadow = (X > 225) & (X < 295) & (Y > 260)
    us[shadow] -= 45.0

    us = np.clip(us, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(us, (3, 3), 0)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = {
        "normal_demo.jpg": generate_normal_cxr(),
        "pneumonia_demo.jpg": generate_pneumonia_cxr(),
        "ct_nodule_demo.jpg": generate_ct_nodule(),
        "mri_tumor_demo.jpg": generate_mri_tumor(),
        "us_breast_demo.jpg": generate_us_breast(),
    }
    for filename, img in samples.items():
        out_path = OUT_DIR / filename
        cv2.imwrite(str(out_path), img)
        print(f"  [OK] {filename} ({img.shape[1]}x{img.shape[0]} - {out_path.stat().st_size} bytes)")

    print(f"\nAll 5 clinical demo samples regenerated successfully at: {OUT_DIR}")


if __name__ == "__main__":
    main()
