# 🐷 Pig Ear Identification System

A deep learning application for **automatic pig identification using ear
images**. The system uses **MobileNetV3-Small** to classify pigs based
on unique ear patterns and provides a **Streamlit web interface** for
easy interaction.

This project helps farmers and researchers **track individual pigs
without physical tags**, using computer vision and machine learning.

------------------------------------------------------------------------

# 📌 Features

-   📷 Upload pig ear images for identification
-   🧠 Deep learning model trained on pig ear dataset
-   🐖 Automatic pig ID prediction
-   📊 Farm dashboard showing registered pigs
-   ⚡ Lightweight model suitable for mobile or edge devices
-   🖥️ Interactive Streamlit interface

------------------------------------------------------------------------

# 🏗️ Project Structure

pig-ear-identification/

├── app.py \# Streamlit application\
├── checkpoints/\
│ ├── best_model.pth \# Trained MobileNetV3 model\
│ └── classes.json \# Class index to Pig ID mapping\
│ ├── dataset/\
│ ├── training/\
│ ├── validation/\
│ └── test/\
│ ├── training/\
│ └── train_model.py \# Model training pipeline\
│ ├── utils/\
│ └── inference.py \# Prediction functions\
│ └── README.md

------------------------------------------------------------------------

# 🧠 Model

Architecture: **MobileNetV3-Small**

Input size: **224 × 224 RGB images**

Loss Function: **CrossEntropyLoss**

Optimizer: **Adam**

Evaluation Metrics:

-   Accuracy
-   Precision
-   Recall
-   F1-score

MobileNetV3-Small was selected because it is:

-   lightweight
-   fast on CPU
-   suitable for mobile deployment

------------------------------------------------------------------------

# 📂 Dataset

Images must follow this naming format:

pig\_\[pig_id\]*IMG*\[image_id\].jpg

Example:

pig_1_IMG_1.jpg\
pig_1_IMG_2.jpg\
pig_5_IMG_1.jpg

Each **pig ID represents a class**.

------------------------------------------------------------------------

# ⚙️ Installation

Clone the repository:

git clone https://github.com/yourusername/pig-ear-identification.git cd
pig-ear-identification

Install dependencies:

pip install -r requirements.txt

Main dependencies:

-   PyTorch
-   torchvision
-   timm
-   streamlit
-   opencv-python
-   scikit-learn

------------------------------------------------------------------------

# 🚀 Running the Application

Start the Streamlit app:

streamlit run app.py

The app will open automatically in your browser.

------------------------------------------------------------------------

# 🐖 Pig Identification Example

Example usage in Python:

from utils.inference import identify_pig

predicted_id, confidence = identify_pig( "test_image.jpg",
model_checkpoint="checkpoints/best_model.pth",
classes_json="checkpoints/classes.json" )

print(predicted_id, confidence)

------------------------------------------------------------------------

# 🏋️ Training the Model

Training uses **MobileNetV3-Small with transfer learning**.

Steps:

1.  Load dataset
2.  Apply image augmentations
3.  Train classifier layer
4.  Validate model
5.  Save best checkpoint

Run training:

python training/train_model.py

The best model will be saved as:

checkpoints/best_model.pth

------------------------------------------------------------------------

# 📊 Evaluation Metrics

After training, the system reports:

-   Accuracy
-   Precision
-   Recall
-   F1 Score

Example:

Accuracy : 92.30%\
Precision : 91.85%\
Recall : 90.76%\
F1-score : 91.20%

------------------------------------------------------------------------

# 🌍 Use Cases

This system can be used for:

-   Precision livestock farming
-   Pig tracking without ear tags
-   Automated farm monitoring
-   Animal research
-   Smart agriculture solutions

------------------------------------------------------------------------

# 🔮 Future Improvements

Possible improvements include:

-   Mobile app integration
-   Real-time camera identification
-   Pig growth tracking
-   Health monitoring
-   Larger dataset training

------------------------------------------------------------------------

# 👨‍💻 Author

Developed by Piggery Research team at Carnegie Mellon University Africa

------------------------------------------------------------------------


