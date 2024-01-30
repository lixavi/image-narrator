import tensorflow as tf
import cv2

def preprocess_image(image_path):
    # Load image using OpenCV
    image = cv2.imread(image_path)
    # Resize image to match input shape of CNN model
    image = cv2.resize(image, (224, 224))
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalize pixel values to range [0, 1]
    image = image / 255.0
    # Expand dimensions to create a batch of size 1
    image = tf.expand_dims(image, axis=0)
    return image
