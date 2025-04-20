# Step 1: Install Gradio (if not already)
!pip install gradio --quiet

# Step 2: Import Libraries
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Step 3: Load Your Model
model = tf.keras.models.load_model('/content/drive/MyDrive/saved_models/model_1.keras')  # Update the path if needed

# Step 4: Preprocess Function
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to match model input
    img = np.array(img) / 255.0   # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Step 5: Prediction Function
def predict(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)[0][0]

    # Assuming binary classification: 0 = Normal, 1 = Osteoporosis
    label = "Osteoporosis" if prediction >= 0.5 else "Normal"
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    return f"Prediction: {label} ({confidence*100:.2f}% confidence)"

# Step 6: Build Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Knee X-ray"),
    outputs=gr.Textbox(label="Diagnosis"),
    title="Osteoporosis Disease Classification",
    description="Upload a knee X-ray image to predict Normal or Osteoporosis"
)

# Step 7: Launch the App
interface.launch(share=True)
