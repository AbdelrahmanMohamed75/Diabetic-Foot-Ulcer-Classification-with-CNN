# 🩺 Diabetic-Foot-Ulcer-with-CNN-




A deep learning project for Diabetic Foot Ulcer (DFU) detection from foot images using Convolutional Neural Networks (CNN) and Transfer Learning.
The system classifies foot images into Normal vs Ulcer, and is deployed as an interactive Streamlit web app for real-time predictions.

🔗 Live App: [https://diabetic-foot-ulcer-classification-with-cnn-2fehf39satqzes7wlm.streamlit.app](https://diabetic-foot-ulcer-classification-with-cnn-2fehf39satqzes7wlm.streamlit.app/)

## 📝 Problem Statement

Diabetic Foot Ulcers are one of the most serious complications of diabetes and can lead to infections or even amputations if not detected early.

Manual diagnosis requires medical expertise and can be time-consuming.
This project aims to build an AI-based classification system that automatically detects DFU from clinical foot images with high accuracy

## 📁 Project Structure 
```
├── Diabetic_Foot_Ulcer_Classification_(DFUC).ipynb   # Main notebook (training + evaluation)
├── app.py                                           # Streamlit deployment file
├── models/                                          # Saved trained model (.keras / .h5)
├── assets/                                          # demo.gif + sample images
├── requirements.txt                                 # Python dependencies
└── README.md                                        # Project documentation

```
---

## 🧩 1: Data Preprocessing & Exploration

- **Dataset Source (Kaggle)**:
  -  https://www.kaggle.com/datasets/laithjj/diabetic-foot-ulcer-dfu?utm_source=chatgpt.com
  
- **Classes**:
  -  Normal (Healthy foot)
  -  Abnormal (Foot with Diabetic Foot Ulcer)
    
- **Preprocessing Steps**:
  -  Resizing images to 224×224
  -  Normalization / scaling
  -  Data augmentation (rotation, flipping, zoom, etc.)
  -  Train/validation split

## 🧠 2: Model Building (CNN & Transfer Learning)

- **Transfer Learning Models Used**:
  -  ResNet
  -  EfficientNet

- **Key Techniques**:
  -  Fine-tuning final layers for DFU classification
  -  Dropout layers to reduce overfitting
  -  Optimizer: Adam
  -  Loss: Binary Crossentropy
  -  Callbacks:
    - EarlyStopping (monitoring validation accuracy)
    - ReduceLROnPlateau

 ## 🧩📊 3: Model Evaluation

 - **Evaluation Metrics**:
   - Accuracy
   - Precision / Recall / F1-score
   - Training vs Validation curves
   
 - **Goal**:
   - Achieve strong performance on unseen images
   - Avoid overfitting using transfer learning + callbacks
## 🚀 4: Deployment (Streamlit)

The final trained model is deployed using Streamlit for real-time predictions.

 - **Input**:
   - Upload a foot image (JPG / PNG)

 - **Output**:
   - Predicted class: Normal / Ulcer
   - Confidence score (%)

 - **🔗 Try it here:**:

https://diabetic-foot-ulcer-classification-with-cnn-2fehf39satqzes7wlm.streamlit.app

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy / Pandas
- Matplotlib / Seaborn
- Scikit-learn
- CNN + Transfer Learning (ResNet, EfficientNet)
- Streamlit

## 🚀 How to Run
1. Clone the repository:

   ```bas
   git clone https://github.com/AbdelrahmanMohamed75/ Diabetic-Foot-Ulcer-Classification-with-CNN.git
   cd  Diabetic-Foot-Ulcer-Classification-with-CNN
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook Diabetic-Foot-Ulcer-Classification-with-CNN.ipynb
   ```
---
4.Try the app: (https://diabetic-foot-ulcer-classification-with-cnn-2fehf39satqzes7wlm.streamlit.app/)

## 👨‍💻 Author
 Made with ❤️ by [Abdelrahman Mohamed Emam]

Feel free to fork ⭐, contribute, or suggest improvements
   
