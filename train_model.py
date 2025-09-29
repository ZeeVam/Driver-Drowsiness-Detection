import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# --- 1. Data Preparation ---

# Define paths
train_dir = 'train'

# Image data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,         # Normalize pixel values to be between 0 and 1
    validation_split=0.2    # Use 20% of the data for validation
)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(24, 24),    # Resize all images to 24x24 pixels
    batch_size=32,
    color_mode='grayscale',  # Use grayscale images for simplicity
    class_mode='binary',     # We have two classes: Open and Closed
    subset='training'        # This is the training set
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(24, 24),
    batch_size=32,
    color_mode='grayscale',
    class_mode='binary',
    subset='validation'    # This is the validation set
)

print("--- DATA SANITY CHECK ---")
print(f"Class Indices Found: {train_generator.class_indices}")
print(f"Total images found for training: {train_generator.n}")
print(f"Total images found for validation: {validation_generator.n}")
print("--------------------------")

# --- 2. Build the CNN Model ---

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    
    Dense(128, activation='relu'),
    Dropout(0.5), # Dropout to prevent overfitting
    
    Dense(1, activation='sigmoid') # Sigmoid for binary classification
])

# --- 3. Compile and Train the Model ---

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Starting model training...")
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=50, 
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32
)

# --- 4. Save the Trained Model ---

print("Training finished. Saving model...")
model.save('drowsiness_model.h5')
print("Model saved as drowsiness_model.h5")