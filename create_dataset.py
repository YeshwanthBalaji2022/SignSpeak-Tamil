# import mediapipe as mp
# import cv2
# import os
# import pickle

# # Initialize MediaPipe hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# DATA_DIR = './selfdata'

# data = []
# labels = []

# for dir_ in os.listdir(DATA_DIR):
#     for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
#         data_aux = []
#         img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         results = hands.process(img_rgb)
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     data_aux.append(x)
#                     data_aux.append(y)

#                 data.append(data_aux)
#                 labels.append(dir_)

# # Save the collected data
# with open('selfdata.pickle', 'wb') as f:
#     pickle.dump({'data': data, 'labels': labels}, f)

import mediapipe as mp
import cv2
import os
import pickle
import numpy as np

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# DATA_DIR = './selfdata'
DATA_DIR = "E:\Millenium\TLFS23 - Tamil Language Finger Spelling Image Dataset\Dataset Folders"
data = []
labels = []

def rotate_coords(coords, angle, center):
    """Rotate coordinates around a center point."""
    angle_rad = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], 
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    rotated_coords = []
    for coord in coords:
        rotated = np.dot(rotation_matrix, coord - center) + center
        rotated_coords.append(rotated)
    return np.array(rotated_coords)

def scale_coords(coords, scale, center):
    """Scale coordinates around a center point."""
    return center + scale * (coords - center)

def translate_coords(coords, translation):
    """Translate coordinates by a certain amount."""
    return coords + translation

def visualize_landmarks(img, coords, color=(0, 255, 0)):
    """Visualize landmarks on the image."""
    for coord in coords:
        x, y = int(coord[0] * img.shape[1]), int(coord[1] * img.shape[0])
        cv2.circle(img, (x, y), 5, color, -1)
    return img

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.append([lm.x, lm.y])
                coords = np.array(coords)

                # Center of the hand landmarks
                center = coords.mean(axis=0)

                # Augmentation: rotation
                for angle in [-10, 0, 10]:  # Rotate by -10, 0, and 10 degrees
                    rotated_coords = rotate_coords(coords, angle, center)
                    data.append(rotated_coords.flatten())
                    labels.append(dir_)

                    # Visualize rotated landmarks
                    img_rotated = visualize_landmarks(img.copy(), rotated_coords)
                    cv2.imshow(f'Rotated {angle} degrees', img_rotated)
                    cv2.waitKey(0)

                # Augmentation: scaling
                for scale in [0.9, 1.0, 1.1]:  # Scale by 0.9x, 1.0x, and 1.1x
                    scaled_coords = scale_coords(coords, scale, center)
                    data.append(scaled_coords.flatten())
                    labels.append(dir_)

                    # Visualize scaled landmarks
                    img_scaled = visualize_landmarks(img.copy(), scaled_coords)
                    cv2.imshow(f'Scaled {scale}x', img_scaled)
                    cv2.waitKey(0)

                # Augmentation: translation
                for tx, ty in [(-0.1, -0.1), (0.0, 0.0), (0.1, 0.1)]:  # Translate by (-0.1, -0.1), (0.0, 0.0), and (0.1, 0.1)
                    translated_coords = translate_coords(coords, np.array([tx, ty]))
                    data.append(translated_coords.flatten())
                    labels.append(dir_)

                    # Visualize translated landmarks
                    img_translated = visualize_landmarks(img.copy(), translated_coords)
                    cv2.imshow(f'Translated ({tx}, {ty})', img_translated)
                    cv2.waitKey(0)

# Ensure data is in numeric format and there are no string types
data = np.array(data, dtype=np.float32)

# Save the collected data
with open('selfdata_augmented.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

cv2.destroyAllWindows()
