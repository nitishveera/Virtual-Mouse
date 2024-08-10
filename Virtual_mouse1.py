import cv2
import numpy as np
import mediapipe as m
from math import sqrt
import pyautogui
import threading

# Initialize MediaPipe hands and drawing utils
m_drawing = m.solutions.drawing_utils
m_hands = m.solutions.hands
click = 0
double_click_counter = 0
video = cv2.VideoCapture(0)

# Smoothing factor for cursor movement
SMOOTHING = 0.5
prev_x, prev_y = 0, 0

# Click debounce threshold
CLICK_DEBOUNCE_THRESHOLD = 10

# Function to move the cursor
def move_cursor(cursor_x, cursor_y):
    pyautogui.moveTo(cursor_x, cursor_y)

# Function to scroll
def scroll(vertical_distance):
    pyautogui.scroll(vertical_distance)

# Function to open the keyboard
def open_keyboard():
    pyautogui.hotkey('win', 'ctrl', 'o')

with m_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while video.isOpened():
        _, frame = video.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        imageHeight, imageWidth, _ = image.shape
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                m_drawing.draw_landmarks(image, hand, m_hands.HAND_CONNECTIONS,
                                         m_drawing.DrawingSpec(color=(250, 0, 0), thickness=2, circle_radius=2))

                # Extract landmark positions
                index_finger_tip = hand.landmark[m_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand.landmark[m_hands.HandLandmark.THUMB_TIP]
                index_finger_tip_x = int(index_finger_tip.x * imageWidth)
                index_finger_tip_y = int(index_finger_tip.y * imageHeight)
                thumb_tip_x = int(thumb_tip.x * imageWidth)
                thumb_tip_y = int(thumb_tip.y * imageHeight)

                # Calculate the distance between the index finger tip and thumb tip
                distance = sqrt((index_finger_tip_x - thumb_tip_x) ** 2 + (index_finger_tip_y - thumb_tip_y) ** 2)

                # Draw a line between index finger tip and thumb tip
                cv2.line(image, (index_finger_tip_x, index_finger_tip_y), (thumb_tip_x, thumb_tip_y), (0, 255, 0), 2)

                # Display the distance
                cv2.putText(image, f'Distance: {int(distance)}', (index_finger_tip_x + 10, index_finger_tip_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Perform a single click if the distance is below a certain threshold
                if distance < 30:
                    click += 1
                    if click % CLICK_DEBOUNCE_THRESHOLD == 0:
                        print("Single click")
                        pyautogui.click()
                        # Draw a circle at the click position
                        cv2.circle(image, (index_finger_tip_x, index_finger_tip_y), 15, (0, 0, 255), 3)

                # Perform a double click if the distance is consistently below a threshold
                if distance < 30:
                    double_click_counter += 1
                    if double_click_counter % (CLICK_DEBOUNCE_THRESHOLD * 2) == 0:
                        print("Double click")
                        pyautogui.doubleClick()
                        # Draw a circle at the double click position
                        cv2.circle(image, (index_finger_tip_x, index_finger_tip_y), 15, (255, 0, 0), 3)

                # Scroll based on the vertical movement of the hand with thumb extended
                if distance < 40:
                    scroll_distance = (thumb_tip_y - index_finger_tip_y) * -2
                    print(f"Scrolling {scroll_distance}")
                    pyautogui.scroll(scroll_distance)

                # Open the keyboard if all five fingers are shown
                if len(hand.landmark) == 21:  # Assuming 21 landmarks for a complete hand
                    extended_fingers = sum([1 for i in [m_hands.HandLandmark.INDEX_FINGER_TIP,
                                                        m_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                                        m_hands.HandLandmark.RING_FINGER_TIP,
                                                        m_hands.HandLandmark.PINKY_TIP]
                                            if hand.landmark[i].y < hand.landmark[i - 2].y])
                    if extended_fingers == 4 and distance > 60:
                        print("Open keyboard")
                        open_keyboard()

                # Move the cursor
                screen_width, screen_height = pyautogui.size()
                cursor_x = int(screen_width * index_finger_tip.x)
                cursor_y = int(screen_height * index_finger_tip.y)

                # Smooth cursor movement
                smoothed_x = prev_x + (cursor_x - prev_x) * SMOOTHING
                smoothed_y = prev_y + (cursor_y - prev_y) * SMOOTHING

                # Update previous cursor position
                prev_x, prev_y = smoothed_x, smoothed_y

                # Use threading to move the cursor
                threading.Thread(target=move_cursor, args=(smoothed_x, smoothed_y)).start()

        # Display the image with landmarks
        cv2.imshow('Virtual Mouse', image)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release video capture and close OpenCV window
video.release()
cv2.destroyAllWindows()
