import cv2
import mediapipe as mp
import pickle
import numpy as np
import createDataset


model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# static image to false here to increase the fluidity of tracking
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.65)
labels_dict = {0: 'Cursor', 1: 'Drawing', 2: 'Select'}

prev_landmarks = None
smoothing_factor = 0.5

while True:
    data_aux = []
    ret, frame, = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
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

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            current_landmarks.extend([x, y])

        if prev_landmarks is not None and len(prev_landmarks) == len(current_landmarks):
            smoothed_landmarks = []
            for i in range(len(current_landmarks)):
                smoothed = prev_landmarks[i] * smoothing_factor + current_landmarks[i] * (1 - smoothing_factor)
                smoothed_landmarks.append(smoothed)
            data_aux = smoothed_landmarks
        else:
            data_aux = current_landmarks

        prev_landmarks = current_landmarks.copy()

        if len(data_aux) == 42:
            normalized_data = createDataset.normalize_hand_landmarks(data_aux)
            prediction = model.predict([np.asarray(normalized_data)])
            predicted_character = labels_dict[int(prediction[0])]
            print(predicted_character)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

