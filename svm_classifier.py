import os
import pandas as pd
import numpy as np
import cv2
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib  # Import joblib for saving the model

# Set your paths
train_dir = 'train'  # Training data directory
test_dir = 'test1'    # Test data directory

# Load training data
def load_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):  # Check for image file types
                    img_path = os.path.join(label_dir, img_file)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: {img_path} could not be read.")
                        continue
                    img = cv2.resize(img, (128, 128))  # Resize images to 128x128
                    images.append(img)
                    labels.append(label)  # Assuming folder name is the label
                else:
                    print(f"Warning: {img_file} is not an image file.")
    return np.array(images), np.array(labels)

# Load the dataset
try:
    print("Loading training data...")
    X, y = load_data(train_dir)
    print(f"Loaded {X.shape[0]} images.")

    # Check the unique classes in the labels
    unique_classes = np.unique(y)
    print(f"Unique classes found: {unique_classes}")
    print(f"Number of classes: {len(unique_classes)}")
    
    if len(unique_classes) <= 1:
        raise ValueError("Not enough classes for training. Please check your training data.")
except Exception as e:
    print(f"Error loading training data: {e}")
    exit()

# Flatten the images for SVM
try:
    X_flat = X.reshape(X.shape[0], -1)  # Flatten images for SVM input
    X_train, X_val, y_train, y_val = train_test_split(X_flat, y, test_size=0.2, random_state=42)

    # Train the SVM model
    print("Training the SVM model...")
    model = svm.SVC(kernel='linear')  # You can change the kernel as needed
    model.fit(X_train, y_train)

    # Validate the model
    y_pred = model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))
except Exception as e:
    print(f"Error during training or validation: {e}")
    exit()

# Save the trained model
try:
    joblib.dump(model, 'svm_model.joblib')  # Save the model to a file
    print("Model saved as svm_model.joblib")
except Exception as e:
    print(f"Error saving the model: {e}")
