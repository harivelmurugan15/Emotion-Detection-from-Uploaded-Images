import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

# Define the transformation used during training to apply to the input image
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Emotion labels (modify if different for your dataset)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load the model
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load("full_model.pth", map_location=device)  # Load the full model
    model.eval()
    return model

model = load_model()

# Function to predict emotion from an image
def predict_emotion(image, model):
    # Transform the image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move image to the same device as the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)
    model = model.to(device)

    # Disable gradients and make a prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_emotion = emotion_labels[predicted.item()]

    return predicted_emotion

# Streamlit app
st.title("Emotion Detection from Images")
st.write("Upload an image to predict the emotion.")

# File uploader
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict the emotion
    st.write("Predicting emotion...")
    predicted_emotion = predict_emotion(image, model)
    st.write(f"**Predicted Emotion:** {predicted_emotion}")
