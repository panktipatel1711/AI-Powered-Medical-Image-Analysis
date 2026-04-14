import cv2
import os
import numpy as np

IMG_SIZE = 224

def load_images(data_dir):
    data = []
    labels = []

    categories = ["NORMAL", "PNEUMONIA"]

    for category in categories:
        path = os.path.join(data_dir, category)
        label = categories.index(category)

        for img in os.listdir(path)[:200]:
            try:
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                data.append(image)
                labels.append(label)
            except:
                pass

    return np.array(data) / 255.0, np.array(labels)