import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Streamlit UI Configuration
st.set_page_config(page_title="Virtual Mouse", layout="wide")
st.title("Virtual Mouse Control with Hand Gestures")
st.markdown("This application uses computer vision to control the mouse using hand gestures.")

# Sidebar for settings
st.sidebar.header("Settings")
hotspot_x = st.sidebar.slider("Hotspot X Position (% of Screen Width):", 0, 20, 10, key="hotspot_x")  # Moved to the left side
hotspot_y = st.sidebar.slider("Hotspot Y Position (% of Screen Height):", 0, 100, 10, key="hotspot_y")
hotspot_size = st.sidebar.slider("Hotspot Size (px):", 50, 200, 100, key="hotspot_size")
smoothing_factor = st.sidebar.slider("Smoothing Factor:", 1, 20, 10, key="smoothing_factor")

# Initialize Mediapipe parameters
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.7)
drawing_utils = mp.solutions.drawing_utils

# State variables for cursor control (in screen coordinates, without pyautogui)
prev_index_x, prev_index_y = 0, 0

# Start Webcam Button
if st.sidebar.button("Start Webcam", key="start_webcam"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the webcam.")
    else:
        st.success("Webcam started successfully. Press 'Stop Webcam' to exit.")

    stop_webcam = st.sidebar.button("Stop Webcam", key="stop_webcam")
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read frame from webcam.")
            break

        # Flip and process the frame
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        # Hand gesture tracking
        if hands:
            for hand_landmarks in hands:
                drawing_utils.draw_landmarks(frame, hand_landmarks)
                landmarks = hand_landmarks.landmark

                # Variables to track fingertip positions
                index_x, index_y = None, None

                for id, landmark in enumerate(landmarks):
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)

                    # Track the index finger tip
                    if id == 8:  # Index finger tip
                        index_x = x
                        index_y = y
                        cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)

                # Cursor movement with smoothing
                if index_x and index_y:
                    index_x = prev_index_x + (index_x - prev_index_x) / smoothing_factor
                    index_y = prev_index_y + (index_y - prev_index_y) / smoothing_factor
                    prev_index_x, prev_index_y = index_x, index_y

                    # Simulate actions based on position
                    hotspot_x_px = frame_width * hotspot_x / 100
                    hotspot_y_px = frame_height * hotspot_y / 100
                    if (hotspot_x_px - hotspot_size < index_x < hotspot_x_px + hotspot_size and
                            hotspot_y_px - hotspot_size < index_y < hotspot_y_px + hotspot_size):
                        st.info("Gesture detected in hotspot area!")

        # Display the frame
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        # Stop Webcam Button
        if stop_webcam:
            break

    cap.release()
    cv2.destroyAllWindows()
    st.success("Webcam stopped successfully.")
