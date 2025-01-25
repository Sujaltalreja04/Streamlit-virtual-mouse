import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Streamlit UI Configuration
st.set_page_config(page_title="Virtual Mouse", layout="wide")
st.title("Virtual Mouse Control with Hand Gestures")
st.markdown("This application uses computer vision to control actions using hand gestures.")

# Sidebar for settings
st.sidebar.header("Settings")
hotspot_x = st.sidebar.slider("Hotspot X Position (% of Screen Width):", 0, 20, 10, key="hotspot_x")
hotspot_y = st.sidebar.slider("Hotspot Y Position (% of Screen Height):", 0, 100, 10, key="hotspot_y")
hotspot_size = st.sidebar.slider("Hotspot Size (px):", 50, 200, 100, key="hotspot_size")
smoothing_factor = st.sidebar.slider("Smoothing Factor:", 1, 20, 10, key="smoothing_factor")

# Initialize Mediapipe parameters
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.7)
drawing_utils = mp.solutions.drawing_utils

# State variables for cursor control (in screen coordinates, without pyautogui)
prev_index_x, prev_index_y = 0, 0
screen_width, screen_height = 800, 600  # Placeholder for screen dimensions

# Webcam Capture using Streamlit's camera_input method
webcam_image = st.camera_input("Capture your webcam feed")

if webcam_image:
    # Convert the webcam image to an OpenCV format
    frame = cv2.imdecode(np.frombuffer(webcam_image.read(), np.uint8), 1)
    
    # Process the image using Mediapipe for hand gesture recognition
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(frame_rgb)
    
    # If hand landmarks are found
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            drawing_utils.draw_landmarks(frame, hand_landmarks)

            # Extract the index finger tip coordinates (id == 8)
            landmarks = hand_landmarks.landmark
            index_x, index_y = None, None

            for id, landmark in enumerate(landmarks):
                if id == 8:  # Index finger tip
                    index_x = int(landmark.x * frame.shape[1])
                    index_y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (index_x, index_y), 10, (0, 255, 255), -1)

            # Track cursor movement (display only in Streamlit)
            if index_x and index_y:
                index_x = prev_index_x + (index_x - prev_index_x) / smoothing_factor
                index_y = prev_index_y + (index_y - prev_index_y) / smoothing_factor
                prev_index_x, prev_index_y = index_x, index_y
                
                # Check if the index finger is in the "hotspot" area
                hotspot_x_px = screen_width * hotspot_x / 100
                hotspot_y_px = screen_height * hotspot_y / 100
                if (hotspot_x_px - hotspot_size < index_x < hotspot_x_px + hotspot_size and
                        hotspot_y_px - hotspot_size < index_y < hotspot_y_px + hotspot_size):
                    st.info("Hand gesture detected in hotspot area!")

    # Display the processed frame
    st.image(frame, channels="BGR", use_column_width=True)
