from src.predict import analyze_medical_image

# 1. Define your paths
image_path = "simple.jpg"
model_path = "medical_model.h5"  # Make sure this file exists in your folder!

# 2. Pass BOTH to the function
result = analyze_medical_image(image_path)