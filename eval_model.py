import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Disable GPU (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load trained model
classifier = load_model('my_model.keras')

# Image Data Augmentation for testing
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the testing dataset
test_set = test_datagen.flow_from_directory(
    'mydata/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Ensure label order is maintained
)

# Evaluate model
test_loss, test_accuracy = classifier.evaluate(test_set, steps=len(test_set))
print(f'Test Accuracy: {test_accuracy:.4f}')

# Get True Labels & Predictions
y_true = test_set.classes  # True class labels
y_pred = np.argmax(classifier.predict(test_set), axis=1)  # Model predictions

# Compute Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Compute overall metrics
total_samples = np.sum(cm)  # Should be 52,000 in your case
total_correct = np.trace(cm)  # Sum of diagonal elements (TP)
total_incorrect = total_samples - total_correct  # Everything else is misclassified

# Compute TP, FP, FN, TN
TP_total = np.sum(np.diag(cm))  # Total correct classifications
FP_total = np.sum(cm, axis=0) - np.diag(cm)  # False positives per class
FN_total = np.sum(cm, axis=1) - np.diag(cm)  # False negatives per class
TN_total = total_samples - (TP_total + FP_total + FN_total)  # Everything else

# Print overall insights
print("\n--- Model Evaluation Summary ---")
print(f"Total Samples: {total_samples}")
print(f"Total Correct Predictions (TP): {total_correct}")
print(f"Total Incorrect Predictions: {total_incorrect}")
print(f"Overall Accuracy: {total_correct / total_samples:.4f}")
print(f"False Positives (FP Total): {np.sum(FP_total)}")
print(f"False Negatives (FN Total): {np.sum(FN_total)}")
print(f"True Negatives (TN Total): {np.sum(TN_total)}")

# Print Classification Report
labels = list(test_set.class_indices.keys())  # Get class labels
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=labels))

# Save & Show Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Overall)')
plt.savefig("confusion_matrix_overall.png")  # Save confusion matrix
plt.show()

