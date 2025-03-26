import os
import mediapipe as mp
import pickle
import cv2

# Function to normalize the data, this way the model is more adaptable to different hand sizes and lighting
def normalize_hand_landmarks(landmarks_list):
    # Extract all x coords on an even index
    x_coords = landmarks_list[0::2]
    # Extract all y coords on an odd index
    y_coords = landmarks_list[1::2]

    # Find the min and max x and y coords
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Calculate the width of the hand in the x direction
    # and the height of the hand in the y direction
    x_range = max_x - min_x
    y_range = max_y - min_y

    # Find the centerpoint of the hand in the x and y directions
    x_center = (min_x + max_x) / 2
    y_center = (min_y + max_y) / 2

    # Add a very small value to prevent division by 0 (will crash) in case
    # the hand isn't moving in a direction
    epsilon = 1e-6
    x_range = max(x_range, epsilon)
    y_range = max(y_range, epsilon)

    # Empty list for the normalized coords
    normalized = []

    # Loop through all the pairs of x and y coords
    for i in range(0, len(landmarks_list), 2):

        # Normalize the x coords by doing this
        # 1. Subtract the center of the hand
        # 2. Divide by half the range (the width of the hand)
        norm_x = (landmarks_list[i] - x_center) / (x_range/2)

        # Normalize the y coords by doing this
        # 1. Subtract the center of hand
        # 2. Divide by half of the range (the height of the hand)
        norm_y = (landmarks_list[i+1] - y_center) / (y_range/2)

        # Add the new normalized data pair to the list
        normalized.extend([norm_x, norm_y])

    # Return the normalized data
    return normalized


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.65)

DATA_DIR = '../data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            normalized_data = normalize_hand_landmarks(data_aux)
            data.append(normalized_data)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
