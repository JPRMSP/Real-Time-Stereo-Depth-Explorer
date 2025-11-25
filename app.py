import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(page_title="Real-Time Stereo Depth Explorer", layout="wide")

st.title("🔵 Real-Time Stereo Depth Explorer (RT-SDE)")
st.write("A dataset-free, model-free real-time 3D reconstruction demo using Stereo Vision.")

# ----------- Utility Functions -----------

def to_gray(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

def compute_sad(left, right, block, disp_range):
    h, w = left.shape
    disp = np.zeros((h, w), np.float32)

    pad = block // 2
    left_pad = np.pad(left, pad)
    right_pad = np.pad(right, pad)

    for y in range(pad, h-pad):
        for x in range(pad, w-pad):
            best = 1e9
            best_d = 0
            L = left_pad[y-pad:y+pad+1, x-pad:x+pad+1]
            for d in range(disp_range):
                xr = x - d
                if xr < pad:
                    continue
                R = right_pad[y-pad:y+pad+1, xr-pad:xr+pad+1]
                cost = np.sum(np.abs(L - R))
                if cost < best:
                    best = cost
                    best_d = d
            disp[y, x] = best_d
    return disp


def compute_zncc(left, right, block, disp_range):
    h, w = left.shape
    disp = np.zeros((h, w), np.float32)

    pad = block // 2
    left_pad = np.pad(left, pad)
    right_pad = np.pad(right, pad)

    for y in range(pad, h-pad):
        for x in range(pad, w-pad):
            best = -1e9
            best_d = 0
            L = left_pad[y-pad:y+pad+1, x-pad:x+pad+1]
            L_mean = np.mean(L)
            L_std = np.std(L) + 1e-5

            for d in range(disp_range):
                xr = x - d
                if xr < pad: 
                    continue
                R = right_pad[y-pad:y+pad+1, xr-pad:xr+pad+1]
                R_mean = np.mean(R)
                R_std = np.std(R) + 1e-5

                zncc = np.sum((L - L_mean) * (R - R_mean)) / (block*block*L_std*R_std)
                if zncc > best:
                    best = zncc
                    best_d = d
            disp[y, x] = best_d

    return disp


def disparity_to_3d(disp):
    h, w = disp.shape
    Q = np.array([
        [1, 0, 0, -w/2],
        [0, -1, 0, h/2],
        [0, 0, 0, -1],
        [0, 0, 1, 0]
    ])

    pts = []
    colors = []

    for y in range(h):
        for x in range(w):
            d = disp[y, x]
            if d > 0:
                X = (x - w/2) / d
                Y = (h/2 - y) / d
                Z = 1.0 / (d + 1e-5)
                pts.append((X, Y, Z))

    pts = np.array(pts)
    return pts


# ------------- UI Layout -----------------

col1, col2 = st.columns(2)

with col1:
    left_img = st.file_uploader("Upload LEFT image", type=["jpg", "png"])
with col2:
    right_img = st.file_uploader("Upload RIGHT image", type=["jpg", "png"])

if left_img and right_img:
    L = Image.open(left_img)
    R = Image.open(right_img)

    st.subheader("Rectified Stereo Pair")
    st.image([L, R], caption=["Left", "Right"], width=300)

    grayL = to_gray(L)
    grayR = to_gray(R)

    algo = st.radio("Select Stereo Algorithm", ["SAD", "ZNCC"])
    block = st.slider("Block Size", 3, 15, 7, step=2)
    disp_range = st.slider("Max Disparity", 5, 60, 25)

    st.subheader("Computing Disparity...")
    if algo == "SAD":
        disp = compute_sad(grayL, grayR, block, disp_range)
    else:
        disp = compute_zncc(grayL, grayR, block, disp_range)

    disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
    st.image(disp_norm, caption="Disparity Map", channels="GRAY")

    st.subheader("3D Reconstruction")
    pts = disparity_to_3d(disp)

    fig = go.Figure(
        data=[go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='markers',
            marker=dict(size=2)
        )]
    )
    fig.update_layout(height=600, scene=dict(aspectmode='data'))
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload both left and right images to begin.")
