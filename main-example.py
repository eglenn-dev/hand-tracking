import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
     # checking whether a hand is detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: # working with each hand
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # cv2.circle(image_object, point_location, radius, color_code, thickness_of_circle)
                colors = [
                    (255, 0, 255),
                    (0, 0, 255),
                    (0, 255, 255),
                    (255, 0, 0),
                    (255, 255, 0),
                    (255, 255, 255),
                    (0, 0, 0),
                    (128, 128, 128),
                    (0, 128, 128),
                    (128, 0, 128),
                    (0, 128, 0),
                    (128, 0, 0),
                    (0, 0, 128),
                    (128, 128, 0),
                    (255, 128, 0),
                    (255, 0, 128),
                    (0, 255, 128),
                    (0, 128, 255),
                    (200, 128, 0),
                    (128, 200, 0),
                    (0, 0, 50)]

                for i in range(len(colors)):
                    if id == i:
                        cv2.circle(image, (cx, cy), 10, colors[i], cv2.FILLED)
                
                

            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

    # Window sizing consistency
    # Get the frame dimensions and calculate the aspect ratio
    height, width, _ = image.shape
    aspect_ratio = width / height

    # Set the window size to maintain the aspect ratio
    window_width = 800  # Set your desired width
    window_height = int(window_width / aspect_ratio)
    
    # Create a resizable window with the calculated size
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video", window_width, window_height)

    cv2.imshow("Video", image)
    cv2.waitKey(1)