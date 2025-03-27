import cv2
import mediapipe as mp
import pickle
import numpy as np
import createDataset


def model():
    # Load model from pickle file
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']

    # Turn on the camera
    cap = cv2.VideoCapture(0)

    # Add the media pipe functions to track the hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # static image to false here to increase the fluidity of tracking
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.65)

    labels_dict = {0: 'Cursor', 1: 'Drawing', 2: 'Select'}

    # Smoothing constants
    prev_landmarks = None
    smoothing_factor = 0.5

    while True:
        # Empty to store the hand landmarks
        data_aux = []

        # Read the data coming in from the camera
        ret, frame, = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frames and extract where the hand is
        results = hands.process(frame_rgb)

        # If the model detects a hand do the following
        if results.multi_hand_landmarks:

            # Draw the landmarks on the hand
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            hand_landmarks = results.multi_hand_landmarks[0]
            current_landmarks = []

            # Get the current x and y position of the hand landmarks and add them to the current landmarks list
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                current_landmarks.extend([x, y])

            # Check to see if the previous landmarks list is populated and if it is use smoothing
            if prev_landmarks is not None and len(prev_landmarks) == len(current_landmarks):
                smoothed_landmarks = []

                # Apply smoothing for better tracking
                for i in range(len(current_landmarks)):
                    smoothed = prev_landmarks[i] * smoothing_factor + current_landmarks[i] * (1 - smoothing_factor)
                    smoothed_landmarks.append(smoothed)
                data_aux = smoothed_landmarks
            else:
                data_aux = current_landmarks

            # Set previous landmarks equal to current landmarks after the smoothing
            prev_landmarks = current_landmarks.copy()

            # If ther are 42 landmarks on screen do the following
            if len(data_aux) == 42:

                # Normalize the landmarks to make it less reliant on lighting, distance or hand size
                normalized_data = createDataset.normalize_hand_landmarks(data_aux)

                # Give the normalized data to the model to classify the gesture being given
                prediction = model.predict([np.asarray(normalized_data)])

                # Assign that classification to a word
                predicted_character = labels_dict[int(prediction[0])]

                # Print what the gesture was classified as
                print(predicted_character)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


model()
