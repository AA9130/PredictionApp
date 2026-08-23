import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree


# Common clothing colors with RGB values
CLOTHING_COLORS = {
    'black'      : (0,   0,   0),
    'white'      : (255, 255, 255),
    'gray'       : (128, 128, 128),
    'light gray' : (211, 211, 211),
    'dark gray'  : (64,  64,  64),
    'red'        : (220, 20,  20),
    'dark red'   : (139, 0,   0),
    'maroon'     : (128, 0,   0),
    'pink'       : (255, 182, 193),
    'hot pink'   : (255, 105, 180),
    'orange'     : (255, 140, 0),
    'yellow'     : (255, 215, 0),
    'beige'      : (245, 245, 220),
    'cream'      : (255, 253, 208),
    'brown'      : (139, 90,  43),
    'dark brown' : (92,  51,  23),
    'tan'        : (210, 180, 140),
    'camel'      : (193, 154, 107),
    'green'      : (34,  139, 34),
    'dark green' : (0,   100, 0),
    'olive'      : (128, 128, 0),
    'mint'       : (152, 255, 152),
    'blue'       : (30,  100, 200),
    'dark blue'  : (0,   0,   139),
    'navy'       : (0,   0,   128),
    'sky blue'   : (135, 206, 235),
    'light blue' : (173, 216, 230),
    'purple'     : (128, 0,   128),
    'violet'     : (148, 0,   211),
    'lavender'   : (230, 190, 255),
    'gold'       : (255, 200, 0),
    'silver'     : (192, 192, 192),
    'teal'       : (0,   128, 128),
    'turquoise'  : (64,  224, 208),
}


def rgb_to_lab(rgb):
    pixel = np.uint8([[list(rgb)]])
    lab   = cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB)
    return lab[0][0].tolist()


def convert_rgb_to_names(rgb_tuple):
    names      = list(CLOTHING_COLORS.keys())
    lab_values = [rgb_to_lab(v) for v in CLOTHING_COLORS.values()]

    kdt_db          = KDTree(lab_values)
    input_lab       = rgb_to_lab(rgb_tuple)
    distance, index = kdt_db.query(input_lab)
    return f'closest match: {names[index]}'

def get_most_prominent_color(image_path, k=3):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # If image has alpha channel (transparent PNG), remove transparent pixels
    if image.shape[2] == 4:
        alpha = image[:, :, 3]
        image = image[:, :, :3]
        opaque_mask = alpha > 10
    else:
        image = image[:, :, :3]
        opaque_mask = np.ones(image.shape[:2], dtype=bool)

    # Filter using HSV — keep pixels with decent saturation and brightness
    # This removes dark shadows, black lining, and washed-out highlights
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value      = hsv[:, :, 2]
    color_mask = (saturation > 30) & (value > 50) & (value < 230)
    mask   = opaque_mask & color_mask
    pixels = image[mask]

    # Fallback if too few pixels pass the filter
    if len(pixels) < 100:
        pixels = image[opaque_mask]

    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)

    centers = kmeans.cluster_centers_

    # Pick the cluster with the highest brightness (avoids dark shadows)
    # but skip near-white clusters (brightness > 220 avg)
    brightness = centers.mean(axis=1)
    valid = brightness < 220
    if valid.any():
        best_cluster = np.argmax(np.where(valid, brightness, -1))
    else:
        best_cluster = np.argmax(brightness)

    # Convert BGR to RGB
    most_prominent_color = centers[best_cluster].astype(int)[::-1]

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