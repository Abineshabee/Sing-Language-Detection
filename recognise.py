import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
classifier = load_model('my_model.keras')

# Define image dimensions
image_x, image_y = 64, 64

# Function to predict letter from an image
def predictor(img_path):
    test_image = image.load_img(img_path, target_size=(image_x, image_y))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)  # Expand dimensions to match model input
    test_image /= 255.0  # Normalize pixel values
    result = classifier.predict(test_image)
    
    # Get predicted class index and probability
    predicted_class = np.argmax(result)
    predicted_prob = result[0][predicted_class]
    
    # Map index to corresponding letter
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    predicted_letter = letters[predicted_class] if predicted_class < len(letters) else '?'
    
    # Print probabilities for all classes
    print("Class Probabilities:")
    for i, prob in enumerate(result[0]):
        print(f"{letters[i]}: {prob:.4f}")
    
    return predicted_letter, predicted_prob

'''
# Manually provide image path
alpha = "abcdefghijklmnopqrstuvwxyz"

for n in range( len( alpha )) :
    image_path = "samples/"+ alpha[n] +".png"  # Change this to your image path
    print("Path of image : ", image_path) 
    predicted_letter, predicted_prob = predictor(image_path)
    print(f"\nPredicted Letter: {predicted_letter} with Probability: {predicted_prob:.4f}")

'''
image_path = "processed_pixels/processed1.png"
predicted_letter, predicted_prob = predictor(image_path)
print(f"\nPredicted Letter: {predicted_letter} with Probability: {predicted_prob:.4f}")

