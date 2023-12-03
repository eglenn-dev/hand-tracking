import cv2
import mediapipe as mp
import math
import os
import pytest

class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5, lmList=[]):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.lmList = lmList

    def handsFinder(self,image,draw=True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positionFinder(self, image, handNo=0):
        if self.results.multi_hand_landmarks:
            self.lmList = []
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id,cx,cy])

    def isThumbsUp(self, lmList=None):
        if lmList == None:
            lmList = self.lmList
        if len(lmList) == 21:
            lengthA = handTracker.calculate_distance(lmList[5], lmList[6]) + handTracker.calculate_distance(lmList[6], lmList[7])
            lengthB = handTracker.calculate_distance(lmList[5], lmList[8])
            if (lengthB < lengthA) and (self.isAbove(lmList[3], [5, 6, 7, 8])) and (self.isAbove(lmList[4], [3, 8])):
                return True
        return False
    
    def isPointingUp(self, lmList=None):
        if lmList == None:
            lmList = self.lmList
        if len(lmList) == 21 and self.handOrientation("up"):
            if (self.isAbove(lmList[7], [4, 12, 16, 20])) and (lmList[8][2] < lmList[6][2]):
                return True
        return False
    
    def isBird(self, lmList=None):
        if lmList == None:
            lmList = self.lmList
        if len(lmList) == 21 and self.handOrientation("up"):
            if (self.isAbove(lmList[10], [4, 8, 16, 20])) and (lmList[12][2] < lmList[10][2]):
                return True
        return False
    
    def isOkay(self, lmList=None):
        if lmList == None:
            lmList = self.lmList
        if len(lmList) == 21 and self.handOrientation("up"):
            difference_index_thumb = handTracker.calculate_distance(lmList[4], lmList[8])
            difference_index_reference = handTracker.calculate_distance(lmList[11], lmList[12])
            if (difference_index_thumb < difference_index_reference * 2) and (not self.isAbove(lmList[6], [12, 16, 20])) and (self.isAbove(lmList[8], [4])) and (self.isAbove(lmList[10], [8])):
                return True
        return False
    
    def isFingerGun(self, lmList=None):
        if lmList == None:
            lmList = self.lmList
        if len(lmList) == 21:
            a = abs(lmList[4][2] - lmList[2][2])
            b = abs(lmList[8][1] - lmList[2][1])
            c = self.calculate_distance(lmList[4], lmList[8])
            angles = handTracker.calculate_triangle_angles(a, b, c)
            distanceA = handTracker.calculate_distance(lmList[5], lmList[8])
            distanceB = handTracker.calculate_distance(lmList[5], lmList[12])
            if (angles[2] > 55 and angles[2] < 100) and self.isAbove(lmList[4], [8]) and self.isAbove(lmList[8], [12, 16, 20]) and (distanceA > distanceB):
                return True
        return False
    
    def isPeace(self, lmList=None):
        if lmList == None:
            lmList = self.lmList
        if len(lmList) == 21:
            distance_knuckles = self.calculate_distance(lmList[6], lmList[10])
            distance_tips = self.calculate_distance(lmList[8], lmList[12])
            distance_index_thumb = self.calculate_distance(lmList[8], lmList[4])
            index_height = self.calculate_distance(lmList[8], lmList[5])
            if (self.isAbove(lmList[8], [5, 9, 13, 16, 17, 20])) and (self.isAbove(lmList[12], [5, 9, 13, 16, 17, 20])) and (distance_index_thumb > index_height) and (distance_tips > (1.1 * distance_knuckles)) and (self.handOrientation("up")) and (self.isAbove(lmList[10], [16, 20])):
                return True
        return False
    
    def isVictory(self, lmList=None):
        if lmList == None:
            lmList = self.lmList
        if len(lmList) == 21 and self.handOrientation("up"):
            distanceA = self.calculate_distance(lmList[10], lmList[14])
            distanceB = self.calculate_distance(lmList[12], lmList[16])
            if (self.isAbove(lmList[16], [10, 14])) and (self.isAbove(lmList[12], [10, 14])) and (distanceB > (1.5 * distanceA)) and self.isAbove(lmList[8], [10, 14]): 
                return True
        return False

    def isO(self, lmList=None):
        if lmList == None:
            lmList = self.lmList
        if len(lmList) == 21 and self.handOrientation("up"):
            average_pos_tips =  self.calculate_average_position([8, 12, 16, 20])
            distance_avg_thumb = self.calculate_distance(average_pos_tips, lmList[4])
            control_distance = self.calculate_distance(lmList[3], lmList[4])
            if (self.isAbove(lmList[6], [5, 9, 13, 17])) and (distance_avg_thumb < control_distance):
                return True
        return False
    
    def isAbove(self, target, landmarks, lmList=None):
        if lmList == None:
            lmList = self.lmList
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

    def handOrientation(self, orientation, lmList=None):
        if lmList == None:
            lmList = self.lmList
        landmarks = list(range(1, 20))
        finger_average_position = self.calculate_average_position(landmarks)
        if len(lmList) == 21:
            if (orientation.lower() == "up"):
                if (finger_average_position[2] < lmList[0][2]):
                    return True
            elif (orientation.lower() == "down"):
                if (finger_average_position[2] > lmList[0][2]):
                    return True
        return False
    
    def handDirection(self, direction, lmList=None):
        if lmList == None:
            lmList = self.lmList
        if len(lmList) == 21:
            average_knuckle_position = self.calculate_average_position([5, 9, 13, 17])
            if direction.lower() == "left":
                if average_knuckle_position[1] > lmList[0][1]:
                    return True
            elif direction.lower() == "right":
                if average_knuckle_position[1] < lmList[0][1]:
                    return True
            else: return False    

    def calculate_average_position(self, landmarks, lmList=None):
        if lmList == None:
            lmList = self.lmList
        x_total = 0
        y_total = 0
        for i in range(len(landmarks)):
            try:
                x_total += lmList[landmarks[i]][1]            
                y_total += lmList[landmarks[i]][2]
            except: return [0, 0, 0]       
        average_x = math.ceil((x_total / len(landmarks)) * 100) / 100
        average_y = math.ceil((y_total / len(landmarks)) * 100) / 100
        return [0, average_x, average_y]
    
    @staticmethod
    def calculate_distance(point1, point2):
        return math.sqrt((point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

    @staticmethod
    def calculate_triangle_angles(a, b, c):
        try:
            # Calculate angles using law of cosines
            A = math.degrees(math.acos((b**2 + c**2 - a**2) / (2 * b * c)))
            B = math.degrees(math.acos((a**2 + c**2 - b**2) / (2 * a * c)))
            C = 180 - A - B
            return A, B, C
        except: return 0, 0, 0

# ======================================================

def main():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()

    if run_pytest('test_handTrackerModule.py'):
        while True:
            success, image = cap.read()
            image = tracker.handsFinder(image)
            tracker.positionFinder(image)
            
            cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
            image = cv2.flip(image, 1)

            # Check for symbols
            clear_console()
            if tracker.isThumbsUp(): print("Thumb up")
            elif tracker.isPointingUp(): print("Pointing up")
            elif tracker.isBird(): print("Bird!")
            elif tracker.isOkay(): print("Okay")
            elif tracker.isFingerGun(): print("Finger gun")
            elif tracker.isPeace(): print("Peace")
            elif tracker.isVictory(): print("Victory")
            elif tracker.isO(): print("O")
            else: print("*No sign detected*")

            # Display updated image
            cv2.imshow("Video",image)

            # Check if the 'q' key is pressed to exit the loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                clear_console()
                print("Keyboard interrupt on 'q' key.")
                break

def clear_console():
    # Check if the operating system is Windows or Unix-based
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def run_pytest(file_path):
    try:
        # Run pytest and capture the result
        result = pytest.main([file_path])

        # Check the result to determine if the tests passed
        if result == 0:
            print("All tests passed!")
            return True
        else:
            print("Some tests failed.")
            return False

    except Exception as e:
        print(f"Error while running pytest: {e}")
        return False

if __name__ == "__main__":
    main()