import cv2
import mediapipe as mp
import pickle
import numpy as np
import os


# Loads the model from the pickle file
def load_model(model_path='./model.p'):
    model_dict = pickle.load(open(model_path, 'rb'))
    return model_dict['model']


# Sets up the mediapipe functions to track the users hand
def setup_mediapipe(static_mode=False, min_confidence=0.65):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=static_mode, min_detection_confidence=min_confidence)

    return mp_hands, mp_drawing, mp_drawing_styles, hands


# Normalize the data so its more adaptable to new hands
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


# Gets the current hand landmarks from the camera and smooths them if possible
def extract_hand_landmarks(results, prev_landmarks=None, smoothing_factor=0.5):
    data_aux = []
    has_sufficient_landmarks = False

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        current_landmarks = []

        # Get the current x and y coords of the hand landmarks
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            current_landmarks.extend([x, y])

        # Apply smoothing if previous landmarks exist
        if prev_landmarks is not None and len(prev_landmarks) == len(current_landmarks):
            smoothed_landmarks = []
            for i in range(len(current_landmarks)):
                smoothed = prev_landmarks[i] * smoothing_factor + current_landmarks[i] * (1 - smoothing_factor)
                smoothed_landmarks.append(smoothed)
            data_aux = smoothed_landmarks
        else:
            data_aux = current_landmarks

        # Update previous landmarks
        prev_landmarks = current_landmarks.copy()

        # Check if we have enough landmarks
        if len(data_aux) == 42:
            has_sufficient_landmarks = True

    return data_aux, prev_landmarks, has_sufficient_landmarks


# Draw the hand landmarks on the users hand
def draw_landmarks(frame, results, mp_hands, mp_drawing, mp_drawing_styles):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    return frame


# Use the model to classify the gesture the user is making
def classify_gesture(model, data, label_dict={0: 'Cursor', 1: 'Drawing', 2: 'Select'}):
    normalized_data = normalize_hand_landmarks(data)
    classification = model.predict([np.asarray(normalized_data)])
    return label_dict[int(classification[0])]


def process_frame(frame, hands, mp_hands, mp_drawing, mp_drawing_styles, model, prev_landmarks, label_dict):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    frame = draw_landmarks(frame, results, mp_hands, mp_drawing, mp_drawing_styles)
    data_aux, prev_landmarks, has_sufficient_landmarks = extract_hand_landmarks(results, prev_landmarks)

    classified_gesture = None

    if has_sufficient_landmarks:
        classified_gesture = classify_gesture(model, data_aux, label_dict)

    return frame, classified_gesture, prev_landmarks


def video_detection(model_path='./model.p', camera_index=0):
    model = load_model(model_path)

    mp_hands, mp_drawing, mp_drawing_styles, hands = setup_mediapipe()

    label_dict = {0: 'Cursor', 1: 'Drawing', 2: 'Select'}

    cap = cv2.VideoCapture(camera_index)

    prev_landmarks = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame, classified_gesture, prev_landmarks = process_frame(
            frame, hands, mp_hands, mp_drawing, mp_drawing_styles,
            model, prev_landmarks, label_dict
        )

        if classified_gesture:
            print(classified_gesture)
            cv2.putText(frame, classified_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_single_image(image_path, model_path='./model.p'):
    model = load_model(model_path)

    mp_hands, mp_drawing, mp_drawing_styles, hands = setup_mediapipe()
    labels_dict = {0: 'Cursor', 1: 'Drawing', 2: 'Select'}

    frame = cv2.imread(image_path)

    frame, predicted_gesture, _ = process_frame(
        frame, hands, mp_hands, mp_drawing, mp_drawing_styles,
        model, None, labels_dict
    )

    return frame, predicted_gesture


def get_gesture_from_image(image, model=None, model_path='./model.p'):
    if model is None:
        model = load_model(model_path)

    mp_hands, mp_drawing, mp_drawing_styles, hands = setup_mediapipe(static_mode=True)
    labels_dict = {0: 'Cursor', 1: 'Drawing', 2: 'Select'}

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    data_aux = []

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.extend([x, y])

        if len(data_aux) == 42:
            return classify_gesture(model, data_aux, labels_dict)

    return None


def create_gesture_detector(model_path='./model.p'):
    model = load_model(model_path)

    def detect_gesture(image):
        return get_gesture_from_image(image, model)

    return {
        'model': model,
        'detect_gesture': detect_gesture
    }


if __name__ == "__main__":
    video_detection()
