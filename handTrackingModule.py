import cv2
import mediapipe as mp
import math
import os

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
        if len(lmList) == 21 and self.handOrientation(lmList, "up"):
            distance_thumb_index = handTracker.calculate_distance(lmList[4], lmList[8])
            distance_index_middle = handTracker.calculate_distance(lmList[8], lmList[12])
            if (distance_thumb_index > (distance_index_middle * 2)) and (lmList[4][2] < lmList[8][2]):
                return True
        return False

    def isPointingUp(self, lmList):
        if len(lmList) == 21 and self.handOrientation(lmList, "up"):
            landmarks = [12, 16, 20]
            average_position = handTracker.calculate_average_position(lmList, landmarks)
            distance_index_middle = handTracker.calculate_distance(lmList[8], lmList[12])
            distance_middle_ring = handTracker.calculate_distance(lmList[12], lmList[16])
            if (distance_index_middle > (distance_middle_ring * 2)) and (lmList[8][2] < average_position[2]) and (lmList[8][2] < lmList[12][2]) and self.isAbove(lmList[8], lmList, [4]):
                return True
        return False
    
    def isBird(self, lmList):
        if len(lmList) == 21 and self.handOrientation(lmList, "up"):
            landmarks = [4, 8, 16, 20]
            if (self.isAbove(lmList[10], lmList, landmarks)) and (lmList[12][2] < lmList[10][2]):
                return True
        return False
    
    def isOkay(self, lmList):
        if len(lmList) == 21 and self.handOrientation(lmList, "up"):
            difference_index_thumb = handTracker.calculate_distance(lmList[4], lmList[8])
            difference_index_reference = handTracker.calculate_distance(lmList[11], lmList[12])
            if (difference_index_thumb < difference_index_reference * 2) and (not self.isAbove(lmList[8], lmList, [12, 16, 20])):
                return True
        return False
    
    def isFingerGun(self, lmList):
        if len(lmList) == 21:
            a = handTracker.calculate_distance(lmList[2], lmList[4])
            b = handTracker.calculate_distance(lmList[2], lmList[8])
            c = handTracker.calculate_distance(lmList[4], lmList[8])
            angles = handTracker.calculate_triangle_angles(a, b, c)

            if angles[2] > 90 and angles[2] < 150 and self.isAbove(lmList[4], lmList, [8]):
                return True
        return False
    
    def isAbove(self, target, lmList, landmarks):
        values = []
        if len(lmList) == 21:
            for lm in landmarks:
                if target[2] < lmList[lm][2]:
                    values.append(True)
                else: values.append(False)
            for val in values:
                if not val:
                    return False
            return True
        return False
    

    def handOrientation(self, lmList, orientation):
        landmarks = list(range(1, 20))
        finger_average_position = handTracker.calculate_average_position(lmList, landmarks)
        if len(lmList) == 21:
            if (orientation.lower() == "up"):
                if (finger_average_position[2] < lmList[0][2]):
                    return True
            elif (orientation.lower() == "down"):
                if (finger_average_position[2] > lmList[0][2]):
                    return True
        return False
    
    def handDirection(self, lmlList, direction):

        if len(lmlList) == 21:
            
            pass
        pass
    
    @staticmethod
    def calculate_average_position(lmList, landmarks):
        x_total = 0
        y_total = 0
        for i in range(len(landmarks)):
            try:
                x_total += lmList[landmarks[i]][1]            
                y_total += lmList[landmarks[i]][2]
            except: return [0, 0, 0]       
        average_x = x_total / len(landmarks)
        average_y = y_total / len(landmarks)
        return [0, average_x, average_y]
    
    @staticmethod
    def calculate_distance(point1, point2):
        return math.sqrt((point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

    @staticmethod
    def calculate_triangle_angles(a, b, c):
        try:
            # Check if the side lengths form a valid triangle
            if a + b <= c or a + c <= b or b + c <= a:
                raise ValueError("Invalid triangle: The sum of any two sides must be greater than the third side.")

            # Calculate the angle opposite side 'a' using the law of cosines
            angle_A_rad = math.acos((b**2 + c**2 - a**2) / (2 * b * c))
            angle_A_deg = math.degrees(angle_A_rad)

            # Calculate the other two angles using the law of sines
            sin_angle_B = (b / c) * math.sin(angle_A_rad)
            angle_B_rad = math.asin(sin_angle_B)
            angle_B_deg = math.degrees(angle_B_rad)

            angle_C_deg = 180 - angle_A_deg - angle_B_deg
        except: return [0, 0, 0]
        return angle_A_deg, angle_B_deg, angle_C_deg

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
        # height, width, _ = image.shape
        # aspect_ratio = width / height

        # Set the window size to maintain the aspect ratio
        # window_width = 1000  # Set your desired width
        # window_height = int(window_width / aspect_ratio)
        
        # Create a resizable window with the calculated size
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Video", window_width, window_height)
        # Mirror the image horizontally
        image = cv2.flip(image, 1)


        # Check for symbols
        clear_console()
        if tracker.isThumbsUp(lmList): print("Thumb up")
        elif tracker.isPointingUp(lmList): print("Pointing up")
        elif tracker.isBird(lmList): print("Bird!")
        elif tracker.isOkay(lmList): print("Okay")
        elif tracker.isFingerGun(lmList): print("Finger gun")
        else: print("*No sign detected*")

        # Display updated image
        cv2.imshow("Video",image)
        cv2.waitKey(1)

def clear_console():
    # Check if the operating system is Windows or Unix-based
    if os.name == 'nt': # Windows
        os.system('cls')
    else:  # for Unix/Linux/MacOS
        os.system('clear')

if __name__ == "__main__":
    main()