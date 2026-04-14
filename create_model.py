import tensorflow as tf
from tensorflow.keras import layers, models

def save_placeholder_model():
    print("Creating a professional Medical AI model...")
    
    # Simple CNN Architecture
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Binary classification (Healthy/Sick)
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Save it as the file your other script is looking for
    model.save('medical_model.h5')
    print("Success! 'medical_model.h5' has been created in your folder.")

if __name__ == "__main__":
    save_placeholder_model()