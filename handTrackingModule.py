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
        # Check to make sure there are 20 points being tracked
        if len(lmList) == 21:
            # Calculating the distance between the thumb and the index finger
            distance_index_thumb = handTracker.calculate_distance(lmList[4], lmList[8])
            # Check if the distance is between 100 and 200
            if (distance_index_thumb > 100) and (distance_index_thumb < 200):
                # Check if the thumb is above the tip of the index finger
                if lmList[4][2] < lmList[8][2]:
                    # Check if the thumb is above the tip of the middle finger
                    if lmList[4][2] < lmList[12][2]:
                        return True
        return False

    def isPointingUp(self, lmList):
        if len(lmList) == 21:
            landmarks = [12, 16, 20]
            average_position = handTracker.calculate_average_position(lmList, landmarks)
            distance = handTracker.calculate_distance(lmList[8], average_position)
            if (lmList[8][2] < average_position[2]) and (distance > 300):
                return True
     
        return False
    
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
        # if len(lmList) != 0:
        #     print(lmList[4])
        if tracker.isThumbsUp(lmList):
            print("Thumb up")
        elif tracker.isPointingUp(lmList):
            print("Pointing up")
            pass
        else: print("no sign detected")
        cv2.imshow("Video",image)
        cv2.waitKey(4)

if __name__ == "__main__":
    main()