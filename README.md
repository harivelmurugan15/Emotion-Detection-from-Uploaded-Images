# Emotion-Detection-from-Uploaded-Images

😄 Emotion Detection from Uploaded Images
🧠 Project Overview
This project is designed to detect human emotions from facial images using a deep learning model (ResNet-18). Users can upload images through a user interface, and the system predicts the emotion (like happy, sad, angry, etc.) present in the image.

🔍 Features
Facial emotion classification using a fine-tuned ResNet-18 model.

Streamlined image upload interface.

Real-time emotion prediction.

Suitable for integration into apps or camera systems.

⚙️ Installation Instructions
Clone the repository:

bash
Copy
Edit
git clone https://github.com/harivelmurugan15/Emotion-Detection-from-Uploaded-Images.git
cd Emotion-Detection-from-Uploaded-Images
Install required packages:

bash
Copy
Edit
pip install -r requirements.txt
Run the user interface:

bash
Copy
Edit
python UserInterface.py
📁 File Structure
UserInterface.py: GUI for uploading and predicting emotion.

model_resnet_18.py: Code to load the pretrained emotion detection model.

kaggle package loader.py: Loads dataset via Kaggle API.

📊 Model Used
ResNet-18: A residual neural network pretrained and fine-tuned for emotion classification.

🧪 Usage
Launch the GUI.

Upload a face image (preferably centered and clear).

Get an instant prediction of the emotion.

🤝 Contributing
Contributions and improvements are welcome! Fork this repo, make your changes, and open a pull request.

📄 License
Licensed under the MIT License.

🙏 Acknowledgments
Dataset: Kaggle FER2013

Libraries used: PyTorch, OpenCV, Tkinter, NumPy
