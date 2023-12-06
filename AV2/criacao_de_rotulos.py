
import os
from Histogram import extract_color_histogram
import numpy as np
def load_data(directory):
    data = []
    labels = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            features = extract_color_histogram(image_path)

            if features is not None:
                label = None

                if "rain" in filename:
                    label = 0
                elif "cloudy" in filename:
                    label = 1
                elif "sunrise" in filename:
                    label = 2
                elif "shine" in filename:
                    label = 3

                if label is not None:
                    data.append(features)
                    labels.append(label)

    return np.array(data), np.array(labels)
