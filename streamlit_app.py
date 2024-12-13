import streamlit as st
import joblib
import cv2
import numpy as np
from skimage import feature
from skimage.transform import resize
from PIL import Image

# Load the model and scaler
clf = joblib.load('model.pkl')  # Replace with your actual model file path
scaler = joblib.load('scaler.pkl')  # Replace with your actual scaler file path


# Define the preprocess_img function (using your method)
def preprocess_img(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)

    resized_img = resize(img,
                         output_shape=(32, 32),
                         anti_aliasing=True)

    hog_feature = feature.hog(
        resized_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm='L2',
        feature_vector=True
    )

    return hog_feature


# Streamlit UI
st.title("Object Detection with Pretrained Model")
st.write("Upload an image to detect objects.")

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    img = Image.open(uploaded_file)
    img = np.array(img)

    # Preprocess and make predictions
    try:
        preprocessed_img = preprocess_img(img)  # Use your custom preprocess_img
        normalized_img = scaler.transform([preprocessed_img])[0]  # Normalize features
        decision = clf.predict_proba([normalized_img])[0]  # Get prediction probabilities

        if np.all(decision < 0.95):
            st.write("No objects detected.")
        else:
            predict_id = np.argmax(decision)
            conf_score = decision[predict_id]
            st.write(f"Predicted Object ID: {predict_id}")
            st.write(f"Confidence Score: {conf_score:.2f}")

        # Display the uploaded image
        st.image(img, caption='Uploaded Image', use_column_width=True)
    except Exception as e:
        st.error(f"An error occurred during inference: {e}")
