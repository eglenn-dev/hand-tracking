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
                    (0, 0, 50)
                    ]

                for i in range(len(colors)):
                    if id == i:
                        cv2.circle(image, (cx, cy), 25, colors[i], cv2.FILLED)
                
                # if id == 9:
                    # cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                    # cv2.circle(image, (cx, cy), 0, (255, 0, 255), 500) 

            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
    cv2.imshow("Output", image)
    cv2.waitKey(1)