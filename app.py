import streamlit as st
import os
import cv2
import torch
import numpy as np

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="2D → VR180 Converter", layout="wide")
st.title("🎬 2D → VR180 Video Converter (CPU-friendly)")
st.write("Upload your 2D MP4 video to generate a VR180 stereoscopic video.")

# -----------------------------
# Upload video
# -----------------------------
uploaded_file = st.file_uploader("Upload a 2D video (MP4)", type=["mp4"])

if uploaded_file is not None:
    # Save uploaded file
    input_video = "input.mp4"
    with open(input_video, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ Video uploaded successfully!")

    # -----------------------------
    # Create folders
    # -----------------------------
    frames_folder = "frames"
    depth_folder = "depth_maps"
    stereo_folder = "stereo_frames"
    output_video = "vr180_output.mp4"
    for folder in [frames_folder, depth_folder, stereo_folder]:
        os.makedirs(folder, exist_ok=True)

    # -----------------------------
    # Extract frames
    # -----------------------------
    st.info("📌 Extracting frames from video...")
    os.system(f'ffmpeg -i "{input_video}" {frames_folder}/frame_%04d.png')

    # -----------------------------
    # Load MiDaS v2.1 Small (CPU)
    # -----------------------------
    st.info("📌 Loading MiDaS v2.1 Small model (CPU-friendly)...")
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
    midas.eval()
    device = torch.device("cpu")
    midas.to(device)

    # Use small_transform
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    # -----------------------------
    # Generate depth maps
    # -----------------------------
    st.info("📌 Generating depth maps...")
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".png")])
    progress_bar = st.progress(0)

    for idx, frame_file in enumerate(frame_files):
        img_path = os.path.join(frames_folder, frame_file)
        depth_path = os.path.join(depth_folder, f"depth_{frame_file}")

        img = cv2.imread(img_path)
        if img is None:
            st.warning(f"⚠️ Frame missing: {img_path}, skipping...")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = transform(img_rgb).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(depth_path, depth_map)

        progress_bar.progress((idx + 1) / len(frame_files))

    # -----------------------------
    # Generate stereoscopic frames (optimized)
    # -----------------------------
    st.info("📌 Generating stereoscopic 3D frames...")
    eye_distance = st.slider("3D Eye Distance (pixels)", 1, 30, 10)

    for frame_file in frame_files:
        img_path = os.path.join(frames_folder, frame_file)
        depth_path = os.path.join(depth_folder, f"depth_{frame_file}")
        out_path = os.path.join(stereo_folder, frame_file)

        img = cv2.imread(img_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

        if img is None or depth is None:
            st.warning(f"⚠️ Skipping {frame_file} (missing image or depth)")
            continue

        depth_norm = depth.astype(np.float32) / 255.0
        shift = (depth_norm * eye_distance).astype(np.int32)

        # Vectorized stereo shift
        h, w = img.shape[:2]
        left_eye = np.zeros_like(img)
        right_eye = np.zeros_like(img)

        x_coords = np.arange(w)
        for y in range(h):
            dx = shift[y, :]
            left_x = np.clip(x_coords - dx, 0, w-1)
            right_x = np.clip(x_coords + dx, 0, w-1)
            left_eye[y, left_x] = img[y, x_coords]
            right_eye[y, right_x] = img[y, x_coords]

        stereo_frame = np.concatenate((left_eye, right_eye), axis=1)
        cv2.imwrite(out_path, stereo_frame)

    # -----------------------------
    # Reassemble VR180 video
    # -----------------------------
    st.info("📌 Reassembling VR180 video...")
    os.system(f'ffmpeg -r 30 -i {stereo_folder}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_video}')

    st.success("✅ VR180 video generated!")
    st.video(output_video)

    # Download button
    with open(output_video, "rb") as f:
        st.download_button("⬇️ Download VR180 Video", f, file_name="vr180_output.mp4")

else:
    st.info("📌 Please upload a 2D MP4 video to start processing.")
