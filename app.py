import streamlit as st
import numpy as np
import cv2
import pywt
from skimage.segmentation import watershed
from scipy import ndimage

st.set_page_config(page_title="Synthetic Lung CT & Lobe Segmentation Simulator", layout="wide")

# -------------------------- UTILITY FUNCTIONS --------------------------

def generate_synthetic_lung(size=256):
    img = np.zeros((size, size), dtype=np.uint8)

    # Left lung
    cv2.ellipse(img, (size//3, size//2), (60, 90), 0, 0, 360, 180, -1)

    # Right lung
    cv2.ellipse(img, (2*size//3, size//2), (70, 100), 0, 0, 360, 180, -1)

    # Add fissure line (horizontal)
    fissure_y = size//2 + 15
    cv2.line(img, (size//4, fissure_y), (3*size//4, fissure_y), 120, 2)

    # Gaussian noise
    noise = np.random.normal(0, 10, (size, size))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img


def adaptive_threshold(img):
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 5
    )


def apply_morphology(binary):
    kernel = np.ones((5,5), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=2)
    return dilated


def watershed_segmentation(binary):
    dist = ndimage.distance_transform_edt(binary)
    markers = ndimage.label(dist > 0.15 * dist.max())[0]
    labels = watershed(-dist, markers, mask=binary)
    return (labels * 50).astype(np.uint8)


def detect_nodules(img):
    # high-intensity spots => possible nodules
    _, th = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
    count = np.sum(th > 0)
    risk = "Low"
    if count > 400: risk = "Medium"
    if count > 900: risk = "High"
    return th, risk


def wavelet_decompose(img):
    coeffs2 = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL, LH, HL, HH

# -------------------------- UI --------------------------

st.title("🫁 Synthetic Lung CT Generator & Lobe Segmentation Simulator")

tab1, tab2, tab3, tab4 = st.tabs(["Generate CT", "Segmentation", "Wavelets", "Cancer Analysis"])

# -------------------------- TAB 1 --------------------------
with tab1:
    st.header("1. Synthetic CT Slice")
    img = generate_synthetic_lung()
    st.image(img, caption="Generated Lung CT Slice", use_column_width=True)

# -------------------------- TAB 2 --------------------------
with tab2:
    st.header("2. Lung Segmentation Pipeline")

    th = adaptive_threshold(img)
    st.subheader("Adaptive Thresholding")
    st.image(th, use_column_width=True)

    morph = apply_morphology(th)
    st.subheader("Morphology (Erosion + Dilation)")
    st.image(morph, use_column_width=True)

    ws = watershed_segmentation(morph)
    st.subheader("Watershed Segmentation (Lobe Simulation)")
    st.image(ws, use_column_width=True)

# -------------------------- TAB 3 --------------------------
with tab3:
    st.header("3. Wavelet Transform")
    LL, LH, HL, HH = wavelet_decompose(img)

    st.subheader("LL (Approximation)")
    st.image(LL, use_column_width=True)

    st.subheader("LH / HL / HH (Detail Coefficients)")
    st.image(np.hstack([LH, HL, HH]), use_column_width=True)

# -------------------------- TAB 4 --------------------------
with tab4:
    st.header("4. Cancer-like Feature Analysis (No ML Used)")

    nod, risk = detect_nodules(img)
    st.subheader(f"Nodule Map — Risk Level: {risk}")
    st.image(nod, use_column_width=True)

    st.markdown("""
    ### Interpretation  
    - Bright regions = high-density anomalies  
    - Rule-based score checks the number of bright pixels  
    - No machine learning or dataset used  
    """)

st.success("App Ready — Deploy via Streamlit, Colab, or GitHub!")
