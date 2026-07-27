import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors
from scipy.spatial import KDTree
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb

def convert_rgb_to_names(rgb_tuple):
    
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return f'closest match: {names[index]}'

def get_most_prominent_color(image_path, k=3):
    image = cv2.imread(image_path)

    pixels = image.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    labels = kmeans.labels_

    unique, counts = np.unique(labels, return_counts=True)

    most_prominent_cluster = unique[np.argmax(counts)]

    most_prominent_color = kmeans.cluster_centers_[most_prominent_cluster].astype(int)

    return most_prominent_color.tolist()

# # Example usage:
# image_path = 'images.jpeg'  # Replace with the path to your image
# most_prominent_color = get_most_prominent_color(image_path)
# print(f"Most Prominent Color (RGB): {tuple(most_prominent_color)}")
# print(convert_rgb_to_names(tuple(most_prominent_color)))

def MainFunction(image):
    image_path = image
    most_prominent_color = get_most_prominent_color(image_path)
    clr = convert_rgb_to_names(tuple(most_prominent_color))
    return clr