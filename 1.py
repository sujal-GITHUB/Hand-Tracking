import cv2
import mediapipe as mp
import time

# Initialize Video Capture object to use the webcam (0 is the default webcam)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize MediaPipe Hands and Drawing Utilities
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cTime = 0
pTime = 0

while True:
    # Capture frame-by-frame
    success, img = cap.read()
    
    # Check if frame is read correctly
    if not success:
        print("Error: Failed to capture image.")
        break

    # Convert the BGR image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the RGB image to detect hands
    results = hands.process(imgRGB)

    # If hands are detected, draw landmarks and connections
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3 )

    # Display the frame
    cv2.imshow("Hand Tracking", img)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
