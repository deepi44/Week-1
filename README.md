🧩 Week 1 – Design Phase Summary

🧠 Problem Statement

The world is rapidly adopting renewable energy sources, with solar power playing a major role in sustainability.
However, the efficiency of solar panels decreases when they are covered with dust, bird droppings, snow, or damaged physically or electrically.
Manual inspection of large solar farms is time-consuming, risky, and inefficient.

To address this issue, the project aims to design an AI-based Solar Panel Condition Classification System that can automatically detect panel conditions using deep-learning and image-processing techniques.

💡 Solution Approach

An automated CNN-based image-classification model is proposed to identify solar-panel conditions accurately and quickly.
The system uses Convolutional Neural Networks (CNNs) to learn image features and classify panels into categories such as:

Clean

Dusty

Bird-drop

Electrical Damage

Physical Damage

Snow-Covered

The model is trained, tested, and evaluated entirely on Google Colab using TensorFlow/Keras with GPU acceleration.

🗂️ Dataset Information

Dataset Name: Solar Panel Images Dataset

Source: Kaggle – Solar Panel Images Dataset

Description: Contains images of solar panels under six different conditions for use in image-classification tasks.

Purpose: Supports AI-based renewable-energy maintenance by enabling automatic detection of panel conditions.

🧱 Design Activities

Collected and explored dataset from Kaggle.

Designed CNN architecture with multiple convolution and pooling layers.

Selected TensorFlow/Keras framework and Google Colab GPU for training.

Planned preprocessing: image resizing (224×224), normalization, and train/test split.

Determined evaluation metrics such as accuracy and loss visualization.

Outcome:
Week 1 successfully completed the system design phase, finalized dataset source, model architecture, and training workflow.

💻 Week 2 – Implementation Phase Summary
⚙️ Implementation Overview

During Week 2, the designed CNN model was implemented and trained using the Solar Panel Image Dataset on Google Colab.
The notebook handled data loading, preprocessing, model training, and result visualization.

🧩 Implementation Steps

1️⃣ Imported the Kaggle dataset into Google Colab using the Kaggle API.
2️⃣ Performed image preprocessing: resizing to 224×224 pixels, normalization, and augmentation.
3️⃣ Built and compiled the CNN model using TensorFlow/Keras with layers:
 - Conv2D + MaxPooling2D (for feature extraction)
 - Flatten + Dense layers (for classification)
 - Dropout (to reduce overfitting)
4️⃣ Trained the model for multiple epochs and plotted accuracy/loss graphs.
5️⃣ Evaluated model performance on validation data and tested with sample solar-panel images.
6️⃣ Saved the trained model (solar_panel_classifier_model.h5) for future testing and deployment.

📊 Results

Training Accuracy: 97.4 %

Validation Accuracy: 95.2 %

Model File: solar_panel_classifier_model.h5

Framework: TensorFlow/Keras on Google Colab

Visualization: Accuracy and loss graphs generated using Matplotlib

Output Example: Predicted label = Dusty 🎯 Accuracy = 97.3 %

🧾 Files Added to GitHub

📘 Solar_Panel_Classifier.ipynb – Google Colab notebook (training and testing)
📄 solar_panel_classifier_model.h5 – Trained CNN model file
📊 accuracy_loss_graph.png – Training/Validation graph
📸 sample_predictions/ – Example classified images
📁 dataset_info.txt – Dataset details and Kaggle source link

✅ Outcome

Week 2 successfully completed the implementation phase — the CNN model achieved high accuracy and demonstrated effective AI-based solar panel condition classification using Google Colab.
This model supports renewable energy management by enabling automatic monitoring of solar panel efficiency and maintenance requirements 🌞⚡
