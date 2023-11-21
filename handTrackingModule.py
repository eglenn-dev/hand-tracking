import cv2
import mediapipe as mp
import math

class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self,image,draw=True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positionFinder(self,image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id,cx,cy])

        return lmlist
    

    def isThumbsUp(self, lmList):
        if len(lmList) == 21:
            landmarks = [8, 12, 16, 20]
            distance_thumb_index = handTracker.calculate_distance(lmList[4], lmList[8])
            distance_index_middle = handTracker.calculate_distance(lmList[8], lmList[12])
            if (distance_thumb_index > (distance_index_middle * 2)):
                return True

        return False

    def isPointingUp(self, lmList):
        if len(lmList) == 21:
            landmarks = [12, 16, 20]
            average_position = handTracker.calculate_average_position(lmList, landmarks)
            distance_index_middle = handTracker.calculate_distance(lmList[8], lmList[12])
            distance_middle_ring = handTracker.calculate_distance(lmList[12], lmList[16])
            if (distance_index_middle > (distance_middle_ring * 2)) and (lmList[8][2] < average_position[2]) and (lmList[8][2] < lmList[12][2]):
                return True
        return False
    
    def isBird(self, lmList):
        if len(lmList) == 21:
            landmarks = [8, 16, 20]
            average_position = handTracker.calculate_average_position(lmList, landmarks)
            distance_index_middle = handTracker.calculate_distance(lmList[8], lmList[12])
            distance_ring_pinky = handTracker.calculate_distance(lmList[16], lmList[20])
            if (distance_index_middle > (distance_ring_pinky * 2)) and (lmList[12][2] < average_position[2]) and (lmList[12][2] < lmList[8][2]):
                return True
        return False
    
    def handOrientation(self, lmList):
        # Function to determine the orientation of the hand
        pass
    
    @staticmethod
    def calculate_average_position(lmList, landmarks):
        x_total = 0
        y_total = 0
        for i in range(len(landmarks)):
            x_total += lmList[landmarks[i]][1]            
            y_total += lmList[landmarks[i]][2]            
        average_x = x_total / len(landmarks)
        average_y = y_total / len(landmarks)
        return [0, average_x, average_y]
    
    @staticmethod
    def calculate_distance(point1, point2):
        return math.sqrt((point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

# ======================================================

def main():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()

    while True:
        success,image = cap.read()
        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image)
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


        # Check for symbols
        if tracker.isThumbsUp(lmList):
            print("Thumb up")
        elif tracker.isPointingUp(lmList):
            print("Pointing up")
            pass
        elif (tracker.isBird(lmList)):
            print("Bird!")
        else: print("no sign detected")

        # Display updated image
        cv2.imshow("Video",image)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()