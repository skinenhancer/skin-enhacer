import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="AI Skin Enhancer", page_icon="✨", layout="wide")

MAX_SIDE = 2048  # bade photos ko resize karte hain taaki app slow na ho


def load_image(file) -> np.ndarray:
    img = Image.open(file)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Skin pixels ka soft mask (0..1) YCrCb color range se."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.GaussianBlur(mask, (0, 0), 7)
    return (mask.astype(np.float32) / 255.0)[..., None]


def enhance_local(bgr, smooth, texture, sharp):
    """OpenCV-based retouch: skin smoothing + texture wapas + sharpening."""
    orig = bgr.astype(np.float32)
    mask = skin_mask(bgr)

    # 1) Smooth (edges bachate hue)
    sigma = 20 + smooth * 80
    smoothed = cv2.bilateralFilter(bgr, d=9, sigmaColor=sigma, sigmaSpace=sigma).astype(np.float32)
    # dobara ek pass strong smoothing ke liye
    if smooth > 0.5:
        smoothed = cv2.bilateralFilter(smoothed.astype(np.uint8), 9, sigma, sigma).astype(np.float32)

    # 2) Fine texture (pores) wapas daalo taaki plastic na lage
    detail = orig - cv2.GaussianBlur(orig, (0, 0), 2.0)
    smoothed = smoothed + detail * texture

    # 3) Sirf skin par apply karo
    out = orig * (1 - mask * smooth) + smoothed * (mask * smooth)

    # 4) Blur removal = unsharp mask
    if sharp > 0:
        blur = cv2.GaussianBlur(out, (0, 0), 1.5)
        out = out + (out - blur) * (sharp * 1.5)

    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_ai(bgr, fidelity):
    """Optional: Replicate par CodeFormer (face restore + deblur)."""
    import replicate  # lazy import

    token = st.secrets.get("REPLICATE_API_TOKEN")
    if not token:
        raise RuntimeError("Secrets me REPLICATE_API_TOKEN add karo.")
    client = replicate.Client(api_token=token)

    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(buf, "PNG")
    buf.seek(0)
    output = client.run(
        "sczhou/codeformer",  # model page par latest version/inputs check kar lena
        input={"image": buf, "codeformer_fidelity": fidelity, "face_upscale": True},
    )
    import requests

    data = requests.get(str(output), timeout=120).content
    arr = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ---------------- UI ----------------
st.title("✨ AI Skin Enhancer")
st.caption("Portrait upload karo — skin smooth aur photo sharp, natural look ke saath.")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Fast (OpenCV, free)", "AI (CodeFormer)"])
    if mode.startswith("Fast"):
        smooth = st.slider("Skin smoothing", 0.0, 1.0, 0.6, 0.05)
        texture = st.slider("Skin texture bachao", 0.0, 1.0, 0.5, 0.05)
        sharp = st.slider("Sharpness / deblur", 0.0, 1.0, 0.3, 0.05)
    else:
        fidelity = st.slider("Fidelity (high = original jaisa)", 0.0, 1.0, 0.7, 0.05)

file = st.file_uploader("Photo upload karo", type=["jpg", "jpeg", "png", "webp"])

if file:
    src = load_image(file)
    with st.spinner("Enhance ho raha hai..."):
        try:
            if mode.startswith("Fast"):
                result = enhance_local(src, smooth, texture, sharp)
            else:
                result = enhance_ai(src, fidelity)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    c1, c2 = st.columns(2)
    c1.subheader("Before")
    c1.image(cv2.cvtColor(src, cv2.COLOR_BGR2RGB), use_container_width=True)
    c2.subheader("After")
    c2.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)

    out_buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)).save(out_buf, "PNG")
    st.download_button("⬇️ Download", out_buf.getvalue(), "enhanced.png", "image/png")
else:
    st.info("Shuru karne ke liye photo upload karo.")
