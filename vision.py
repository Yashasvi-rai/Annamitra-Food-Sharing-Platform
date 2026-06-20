import cv2
import numpy as np

MAX_CONTAINER_WEIGHT = 5.0  # kg


def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Invalid image path")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    return img, blur


def detect_largest_contour(binary_img):
    contours, _ = cv2.findContours(
        binary_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No contours detected")

    largest = max(contours, key=cv2.contourArea)
    return largest


def get_container_height(container_image_path):
    img, processed = preprocess_image(container_image_path)

    edges = cv2.Canny(processed, 50, 150)

    contour = detect_largest_contour(edges)

    x, y, w, h = cv2.boundingRect(contour)

    return h


def get_food_height(food_image_path):
    img, processed = preprocess_image(food_image_path)

    edges = cv2.Canny(processed, 50, 150)

    contour = detect_largest_contour(edges)

    x, y, w, h = cv2.boundingRect(contour)

    return h


def estimate_weight(container_image_path, food_image_path):
    container_height = get_container_height(container_image_path)
    food_height = get_food_height(food_image_path)

    if container_height == 0:
        raise ValueError("Container height detection failed")

    fill_ratio = food_height / container_height

    # clamp ratio between 0 and 1
    fill_ratio = max(0, min(fill_ratio, 1))

    estimated_weight = round(MAX_CONTAINER_WEIGHT * fill_ratio, 2)

    return {
        "container_height_px": container_height,
        "food_height_px": food_height,
        "fill_ratio": round(fill_ratio, 3),
        "estimated_weight_kg": estimated_weight
    }
