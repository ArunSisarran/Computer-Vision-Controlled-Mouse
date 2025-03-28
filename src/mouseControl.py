import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Button, Controller
import time


def index_finger_control():

    # Initialize mouse controller
    mouse = Controller()

    # Setting up mediapipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.7)

    # Get screen dimensions - hardcoded since we can't use pyautogui.size()
    # You may need to adjust these values for your screen
    screen_width, screen_height = 1920, 1080  

    # Start camera
    camera = cv2.VideoCapture(0)

    # Smoothing variables
    previous_x, previous_y = 0, 0
    smoothing_factor = 0.5

    # ID's of the index and thumb landmarks
    thumb = 4
    index = 8
    selection_mode = False
    selection_timer = 0
    selection_threshold = 20

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Flip the camera so things are not inverted
        frame = cv2.flip(frame, 1)

        frame_height, frame_width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        # variables of where the mouse cursor is
        cursor_x, cursor_y = 0, 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Getting the index finger postion for the cursor
                index_finger = hand_landmarks.landmark[index]
                cursor_x = int(index_finger.x * frame_width)
                cursor_y = int(index_finger.y * frame_height)

                # Getting the thumb position for selecting
                thumb_finger = hand_landmarks.landmark[thumb]
                thumb_x = int(thumb_finger.x * frame_width)
                thumb_y = int(thumb_finger.y * frame_height)

                # Distance between the index and the thumb
                distance = np.sqrt((cursor_x - thumb_x) ** 2 + (cursor_y - thumb_y) ** 2)

                # Draws a circle on the index finger
                cv2.circle(frame, (cursor_x, cursor_y), 10, (0, 255, 0), -1)

                # Apply smoothing to cursor movement
                if previous_x > 0 and previous_y > 0:
                    cursor_x = int(previous_x * smoothing_factor + cursor_x * (1 - smoothing_factor))
                    cursor_y = int(previous_y * smoothing_factor + cursor_y * (1 - smoothing_factor))
                
                # Update previous positions
                previous_x, previous_y = cursor_x, cursor_y
                
                # Map index finger position to screen coordinates
                mouse_x = int(cursor_x * screen_width / frame_width)
                mouse_y = int(cursor_y * screen_height / frame_height)
                
                # Move mouse cursor
                mouse.position = (mouse_x, mouse_y)

                # Detection if you are pinching your thumb and index so it goes into selection mode
                if distance < 40:
                    cv2.putText(frame, "Clicking", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    selection_timer += 1

                    # Progress bar on the clicking action
                    progress = min(selection_timer / selection_threshold, 1.0)
                    bar_width = int(100 * progress)
                    cv2.rectangle(frame, (10, 70), (10 + bar_width, 85), (0, 0, 255), -1)

                    # If you reach that threshold click
                    if selection_timer >= selection_threshold and not selection_mode:
                        mouse.press(Button.left)
                        mouse.release(Button.left)
                        selection_mode = True

                else:
                    selection_timer = 0
                    selection_mode = False

        # Display the frame
        cv2.putText(frame, "Press 'q' to quit", (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Index Finger Mouse Control', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

    # Release resources
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    time.sleep(1)
    index_finger_control()
