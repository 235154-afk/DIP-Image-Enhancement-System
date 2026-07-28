"""
Smart Image Enhancement & Analysis System
Muhammad Talal Tariq | 235154 | Python + OpenCV

Refactored version — each phase is isolated into its own function
(no globals shared across phases except explicit arguments/returns).

NOTE ON RUNNING THIS FILE:
    This is a plain .py script, not a notebook. `!pip install ...` is
    Jupyter/Colab-only syntax and is INVALID here. Install dependencies
    from a terminal before running:

        pip install opencv-python-headless matplotlib numpy

    (In a Colab/Jupyter notebook you can still use `!pip install ...`
    in its own cell — just don't put that line inside a .py file.)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────

def save_fig(fig, name, output_dir="outputs"):
    """Save a matplotlib figure to output_dir, show it, then close it."""
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f"{output_dir}/{name}", dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def load_image(image_path):
    """Load an image once and derive the RGB / grayscale versions used
    by every later phase. Returns a plain dict so each phase function
    takes explicit inputs instead of reaching for globals."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w, c = img_rgb.shape

    return {
        "bgr": img_bgr,
        "rgb": img_rgb,
        "gray": img_gray,
        "h": h,
        "w": w,
        "c": c,
    }


# ──────────────────────────────────────────────────────────────
# Phase 6.1 — Image Acquisition
# ──────────────────────────────────────────────────────────────

def phase_acquisition(image_data, output_dir="outputs"):
    img_rgb = image_data["rgb"]
    img_gray = image_data["gray"]
    h, w, c = image_data["h"], image_data["w"], image_data["c"]

    print("=" * 55)
    print("  PHASE 6.1 : Image Acquisition & Understanding")
    print("=" * 55)
    print(f"  Resolution : {h} x {w}")
    print(f"  Channels   : {c}")
    print(f"  Data Type  : {img_rgb.dtype}")
    print("  Pixel Matrix (top-left 5x5 grayscale):")
    print(img_gray[:5, :5])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Phase 6.1 - Image Acquisition & Understanding", fontsize=13, fontweight="bold")

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original RGB")
    axes[0].axis("off")

    axes[1].imshow(img_gray, cmap="gray")
    axes[1].set_title("Grayscale")
    axes[1].axis("off")

    axes[2].imshow(img_gray[:30, :30], cmap="gray")
    axes[2].set_title("Pixel Matrix (30x30)")
    axes[2].axis("off")

    fig.text(
        0.5,
        -0.01,
        f"Resolution:{h}x{w} | Channels:{c} | dtype:{img_rgb.dtype}",
        ha="center",
        fontsize=10,
        color="navy",
    )
    save_fig(fig, "phase61_acquisition.jpg", output_dir)


# ──────────────────────────────────────────────────────────────
# Phase 6.2 — Sampling & Quantization
# ──────────────────────────────────────────────────────────────

def phase_sampling(image_data, output_dir="outputs"):
    img_gray = image_data["gray"]
    h, w = image_data["h"], image_data["w"]

    print("\n" + "=" * 55)
    print("  PHASE 6.2 : Sampling & Quantization")
    print("=" * 55)

    scales = [0.25, 0.5, 1.0, 1.5, 2.0]
    scale_labels = ["0.25x (Down)", "0.50x (Down)", "1x (Original)", "1.50x (Up)", "2x (Up)"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle("Phase 6.2A - Sampling Comparison", fontsize=13, fontweight="bold")

    for i, scale in enumerate(scales):
        new_h, new_w = int(h * scale), int(w * scale)
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(img_gray, (new_w, new_h), interpolation=interpolation)

        axes[i].imshow(resized, cmap="gray")
        axes[i].set_title(f"{scale_labels[i]}\n{new_h}x{new_w}")
        axes[i].axis("off")
        print(f"  {scale_labels[i]:15} -> {new_h} x {new_w}")

    save_fig(fig, "phase62a_sampling.jpg", output_dir)


def phase_quantization(image_data, output_dir="outputs"):
    img_gray = image_data["gray"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Phase 6.2B - Bit Depth Reduction", fontsize=13, fontweight="bold")

    notes = ["Full quality", "Slight banding", "Heavy posterisation"]
    for i, bit_depth in enumerate([8, 4, 2]):
        levels = 2 ** bit_depth
        quantized = np.uint8(np.round(img_gray / 255 * (levels - 1)) * (255 / (levels - 1)))

        axes[i].imshow(quantized, cmap="gray")
        axes[i].set_title(f"{bit_depth}-bit ({levels} levels)\n{notes[i]}")
        axes[i].axis("off")
        print(f"  {bit_depth}-bit -> {levels} levels -> {notes[i]}")

    save_fig(fig, "phase62b_quantization.jpg", output_dir)


# ──────────────────────────────────────────────────────────────
# Phase 6.3 — Geometric Transformations
# ──────────────────────────────────────────────────────────────

def phase_rotation(image_data, output_dir="outputs"):
    img_gray = image_data["gray"]
    h, w = image_data["h"], image_data["w"]
    cx, cy = w // 2, h // 2

    print("\n" + "=" * 55)
    print("  PHASE 6.3 : Geometric Transformations")
    print("=" * 55)

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    fig.suptitle("Phase 6.3A - Rotation (7 Angles)", fontsize=13, fontweight="bold")
    axes = axes.flatten()

    axes[0].imshow(img_gray, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, angle in enumerate([30, 45, 60, 90, 120, 150, 180]):
        rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(img_gray, rotation_matrix, (w, h))

        axes[i + 1].imshow(rotated, cmap="gray")
        axes[i + 1].set_title(f"Rotation {angle} deg")
        axes[i + 1].axis("off")

    save_fig(fig, "phase63a_rotation.jpg", output_dir)


def phase_translation_and_shear(image_data, output_dir="outputs"):
    img_gray = image_data["gray"]
    h, w = image_data["h"], image_data["w"]

    translation_matrix = np.float32([[1, 0, 50], [0, 1, 30]])
    translated = cv2.warpAffine(img_gray, translation_matrix, (w, h))

    src_points = np.float32([[0, 0], [w, 0], [0, h]])
    dst_points = np.float32([[0, 0], [w, 0], [int(0.3 * h), h]])
    shear_matrix = cv2.getAffineTransform(src_points, dst_points)
    sheared = cv2.warpAffine(img_gray, shear_matrix, (w, h))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Phase 6.3B - Translation & Shearing", fontsize=13, fontweight="bold")

    images = [img_gray, translated, sheared]
    titles = ["Original", "Translated (50,30)", "Horizontal Shear"]
    for ax, im, title in zip(axes, images, titles):
        ax.imshow(im, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    save_fig(fig, "phase63b_trans_shear.jpg", output_dir)


def phase_inverse_transform(image_data, output_dir="outputs"):
    img_gray = image_data["gray"]
    h, w = image_data["h"], image_data["w"]
    cx, cy = w // 2, h // 2

    forward_matrix = cv2.getRotationMatrix2D((cx, cy), 90, 1.0)
    inverse_matrix = cv2.getRotationMatrix2D((cx, cy), -90, 1.0)

    rotated_90 = cv2.warpAffine(img_gray, forward_matrix, (w, h))
    restored = cv2.warpAffine(rotated_90, inverse_matrix, (w, h))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Phase 6.3C - Inverse Transformation", fontsize=13, fontweight="bold")

    images = [img_gray, rotated_90, restored]
    titles = ["Original", "Rotated 90 deg", "Restored (-90 deg)"]
    for ax, im, title in zip(axes, images, titles):
        ax.imshow(im, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    save_fig(fig, "phase63c_inverse.jpg", output_dir)

    print("  Translation : tx=50, ty=30 (done)")
    print("  Shearing    : Horizontal factor=0.3 (done)")
    print("  Inverse     : Rotate 90 -> Restore -90 (done)")


# ──────────────────────────────────────────────────────────────
# Phase 6.4 — Intensity Transformations
# ──────────────────────────────────────────────────────────────

def phase_intensity_transformations(image_data, output_dir="outputs"):
    img_gray = image_data["gray"]

    print("\n" + "=" * 55)
    print("  PHASE 6.4 : Intensity Transformations")
    print("=" * 55)

    normalized = img_gray.astype(np.float64) / 255.0
    negative = 1.0 - normalized

    log_transformed = np.log1p(normalized)
    log_transformed /= log_transformed.max()

    gamma_bright = np.power(normalized, 0.5)   # gamma < 1 -> brighter
    gamma_dark = np.power(normalized, 1.5)     # gamma > 1 -> darker

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle("Phase 6.4 - Intensity Transformations", fontsize=13, fontweight="bold")

    images = [normalized, negative, log_transformed, gamma_bright, gamma_dark]
    titles = ["Original", "Negative", "Log (C=1)", "Gamma=0.5\n(Brighter)", "Gamma=1.5\n(Darker)"]
    for ax, im, title in zip(axes, images, titles):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    save_fig(fig, "phase64_intensity.jpg", output_dir)

    print("  Gamma=0.5 -> BEST for BRIGHTENING")
    print("  Log       -> BEST for dark detail highlighting")


# ──────────────────────────────────────────────────────────────
# Phase 6.5 — Histogram Processing
# ──────────────────────────────────────────────────────────────

def compute_histogram_equalization_lut(img_gray):
    """Isolated so it can be reused/tested on its own (used again in
    the final pipeline in phase_final_pipeline)."""
    counts, _ = np.histogram(img_gray.ravel(), 256, (0, 256))
    pdf = counts / counts.sum()
    cdf = np.cumsum(pdf)
    lut = np.uint8(255 * cdf)
    return lut, cdf


def phase_histogram_processing(image_data, output_dir="outputs"):
    img_gray = image_data["gray"]

    print("\n" + "=" * 55)
    print("  PHASE 6.5 : Histogram Processing")
    print("=" * 55)

    counts, _ = np.histogram(img_gray.ravel(), 256, (0, 256))
    active_bins = np.sum(counts > 0)
    contrast_label = "LOW contrast" if active_bins < 100 else "HIGH contrast"
    print(f"  Active bins: {active_bins}/256 -> {contrast_label}")

    lut, cdf = compute_histogram_equalization_lut(img_gray)
    img_eq_manual = lut[img_gray]
    img_eq_builtin = cv2.equalizeHist(img_gray)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Phase 6.5 - Histogram Equalization Comparison", fontsize=13, fontweight="bold")

    images = [img_gray, img_eq_manual, img_eq_builtin]
    titles = ["Original", "Manual HE", "cv2.equalizeHist()"]
    for col, (im, title) in enumerate(zip(images, titles)):
        axes[0][col].imshow(im, cmap="gray")
        axes[0][col].set_title(title)
        axes[0][col].axis("off")

        axes[1][col].hist(im.ravel(), bins=256, range=(0, 255), color="steelblue", edgecolor="none")
        axes[1][col].set_title(f"Histogram: {title}")
        axes[1][col].set_xlabel("Intensity")

    save_fig(fig, "phase65_histogram.jpg", output_dir)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cdf, color="blue", linewidth=2)
    ax.set_title("CDF Used for Equalization")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Cumulative Probability")
    ax.grid(True)
    save_fig(fig, "phase65_cdf.jpg", output_dir)

    std_original = np.std(img_gray.astype(float))
    std_manual = np.std(img_eq_manual.astype(float))
    std_builtin = np.std(img_eq_builtin.astype(float))

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Original", "Manual HE", "equalizeHist()"]
    values = [std_original, std_manual, std_builtin]
    bars = ax.bar(labels, values, color=["#4472C4", "#ED7D31", "#A9D18E"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{value:.1f}", ha="center")
    ax.set_ylabel("Standard Deviation")
    ax.set_title("Contrast Comparison (Std Dev)")
    ax.grid(axis="y")
    save_fig(fig, "phase65_stddev.jpg", output_dir)

    print(f"  Std Dev -> Original:{std_original:.1f}  ManualHE:{std_manual:.1f}  cv2HE:{std_builtin:.1f}")


# ──────────────────────────────────────────────────────────────
# Phase 6.6 — Final Integrated Pipeline
# ──────────────────────────────────────────────────────────────

def enhance_image(img_bgr):
    """The actual reusable enhancement pipeline: Grayscale -> Log ->
    Gamma(0.5) -> Histogram Equalization. Kept independent of any
    plotting/printing so it can be called directly by other code
    (e.g. a Streamlit app, or a unit test) without side effects."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr.copy()

    normalized = gray.astype(np.float64) / 255.0
    log_transformed = np.log1p(normalized)
    log_transformed = (log_transformed / log_transformed.max() * 255).astype(np.uint8)

    gamma_corrected = np.power(log_transformed.astype(np.float64) / 255.0, 0.5)
    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)

    lut, _ = compute_histogram_equalization_lut(gamma_corrected)
    return lut[gamma_corrected]


def phase_final_pipeline(image_data, output_dir="outputs"):
    img_bgr = image_data["bgr"]
    img_rgb = image_data["rgb"]
    img_gray = image_data["gray"]

    print("\n" + "=" * 55)
    print("  PHASE 6.6 : Final Integrated Pipeline")
    print("=" * 55)

    enhanced = enhance_image(img_bgr)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Phase 6.6 - Final Pipeline Result", fontsize=13, fontweight="bold")

    axes[0].imshow(img_rgb)
    axes[0].set_title("INPUT: Original RGB")
    axes[0].axis("off")

    axes[1].imshow(img_gray, cmap="gray")
    axes[1].set_title("Step1: Grayscale")
    axes[1].axis("off")

    axes[2].imshow(enhanced, cmap="gray")
    axes[2].set_title("OUTPUT: Enhanced")
    axes[2].axis("off")

    save_fig(fig, "phase66_pipeline.jpg", output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Before vs After Histogram", fontsize=13, fontweight="bold")

    axes[0].hist(img_gray.ravel(), bins=256, range=(0, 255), color="steelblue", edgecolor="none")
    axes[0].set_title("Before")
    axes[0].set_xlabel("Intensity")

    axes[1].hist(enhanced.ravel(), bins=256, range=(0, 255), color="tomato", edgecolor="none")
    axes[1].set_title("After (Enhanced)")
    axes[1].set_xlabel("Intensity")

    save_fig(fig, "phase66_before_after_hist.jpg", output_dir)

    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/235154_enhanced_output.jpg"
    cv2.imwrite(output_path, enhanced)

    print("  Pipeline: Grayscale -> Log -> Gamma(0.5) -> HistEQ (done)")
    print(f"  Saved -> {output_path} (done)")

    return enhanced


# ──────────────────────────────────────────────────────────────
# Q&A summary (kept as a plain function, no globals needed)
# ──────────────────────────────────────────────────────────────

def print_qa_summary():
    print("\n" + "=" * 55)
    print("  Q&A - Section 7.4")
    print("=" * 55)
    print("Q1: HE spreads intensities across full 0-255 range using CDF -> improves contrast.")
    print("Q2: Gamma<1 brightens (raises dark pixels). Gamma>1 darkens (compresses brights).")
    print("Q3: Fewer bit levels = banding/posterisation = visible quality loss.")
    print("Q4: Rotation is reversible (+90 -> -90 = original). Quantization is NOT reversible.")
    print("Q5: Geometric transforms shift pixel positions. Intensity transforms change pixel values.")


# ──────────────────────────────────────────────────────────────
# Orchestration — this is the only place phases are chained together
# ──────────────────────────────────────────────────────────────

def run_pipeline(image_path, output_dir="outputs"):
    image_data = load_image(image_path)

    phase_acquisition(image_data, output_dir)
    phase_sampling(image_data, output_dir)
    phase_quantization(image_data, output_dir)
    phase_rotation(image_data, output_dir)
    phase_translation_and_shear(image_data, output_dir)
    phase_inverse_transform(image_data, output_dir)
    phase_intensity_transformations(image_data, output_dir)
    phase_histogram_processing(image_data, output_dir)
    phase_final_pipeline(image_data, output_dir)
    print_qa_summary()

    print("\n" + "*" * 55)
    print("  ALL 6 PHASES COMPLETE")
    print(f"  All figures saved in -> {output_dir}/ folder")
    print("*" * 55)


if __name__ == "__main__":
    IMAGE_PATH = "hawkes_bay_in (1).jpg"
    run_pipeline(IMAGE_PATH)
