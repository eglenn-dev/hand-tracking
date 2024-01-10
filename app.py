import os
import cv2
import pytest
from handTrackingModule import handTracker

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
            print('To exit, press the \'q\' key.\n')
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