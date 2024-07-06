import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# Ensure the file path is correct
model_path = r"E:\Brain Tumor Classification\model.keras"
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    # Load the trained model
    model = load_model(model_path)

    # Define the class names
    class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

    # Set the target image size
    image_size = (128, 128)

    # Function to preprocess the uploaded image
    def preprocess_image(image):
        image = image.resize(image_size)
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = image.astype('float32') / 255.0
        return image

    # Streamlit app title and description
    st.title("Brain Tumor Classification")
    st.write("Upload an MRI image of a brain to classify if it has a tumor and what type of tumor it is.")

    # File uploader for image
    uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Predict the class of the image
        prediction = model.predict(processed_image)
        predicted_class = class_names[np.argmax(prediction)]
        
        # Display the prediction result
        st.subheader(f"The model predicts that the tumor type is: {predicted_class}")
        
        # Display prediction probabilities
        st.subheader("Prediction Probabilities:")
        probabilities = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
        st.write(probabilities)
        
        # Display the image with matplotlib (optional)
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.set_title(f"Predicted: {predicted_class}")
        st.pyplot(fig)
