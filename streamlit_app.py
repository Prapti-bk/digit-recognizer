"""Streamlit UI. Presentation only: all logic lives in the digit_recognizer package."""
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from digit_recognizer.config import DEFAULT_MODEL_PATH
from digit_recognizer.errors import BlankImageError, ModelNotFoundError
from digit_recognizer.predictor import DigitPredictor

st.set_page_config(page_title="Digit Recognizer", page_icon="✍️")
st.title("✍️ Handwritten Digit Recognizer")
st.caption("A CNN trained on MNIST. Draw a single digit (0-9) or upload an image.")


@st.cache_resource
def get_predictor() -> DigitPredictor:
    return DigitPredictor.from_path(DEFAULT_MODEL_PATH)


def canvas_to_image(rgba_array) -> Image.Image:
    """Flatten the canvas's transparent RGBA strokes onto a black background."""
    rgba = Image.fromarray(rgba_array.astype("uint8"), "RGBA")
    background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    background.alpha_composite(rgba)
    return background.convert("L")


try:
    predictor = get_predictor()
except ModelNotFoundError as exc:
    st.error(str(exc))
    st.stop()

draw_tab, upload_tab = st.tabs(["Draw", "Upload image"])
image = None

with draw_tab:
    canvas = st_canvas(stroke_width=18, stroke_color="#FFFFFF", background_color="#000000",
                       height=280, width=280, drawing_mode="freedraw", key="canvas")
    if canvas.image_data is not None:
        image = canvas_to_image(canvas.image_data)

with upload_tab:
    uploaded = st.file_uploader("PNG/JPG of a single digit", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, width=140)

if image is not None:
    try:
        result = predictor.predict(image)
    except BlankImageError:
        st.info("Draw a digit to see a prediction.")
    else:
        left, right = st.columns([1, 2])
        left.metric("Prediction", result.digit, f"{result.confidence:.1%} confident")
        left.image(result.model_input, caption="What the model sees", width=112, clamp=True)
        right.bar_chart(pd.DataFrame({"probability": result.probabilities}))
