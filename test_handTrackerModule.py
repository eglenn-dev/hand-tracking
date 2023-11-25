import cv2
import mediapipe as mp
import pytest
from handTrackingModule import handTracker

lmList1 = [[0, 354, 485], [1, 302, 456], [2, 266, 407], [3, 242, 366], [4, 218, 336], [5, 314, 335], [6, 298, 279], [7, 287, 245], [8, 278, 214], [9, 350, 330], [10, 353, 268], [11, 351, 228], [12, 349, 194], [13, 380, 341], [14, 392, 282], [15, 397, 245], [16, 398, 211], [17, 406, 361], [18, 421, 319], [19, 429, 289], [20, 435, 260]]
lmList2 = []

# Create a fixture to initialize the handTracker class for testing
@pytest.fixture
def hand_tracker_instance():
    return handTracker()

def test_general():
    assert check_array_for_values(lmList1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    assert check_array_for_values(lmList1, [19, 18, 14, 15, 2, 0])

def check_array_for_values(lmList, landmarks):
    values = []
    for item in landmarks:
        if item in lmList[item]: values.append(True)
        else: values.append(False)
    for val in values:
        if not val:
            return False
    return True


pytest.main(["-v", "--tb=line", "-rN", __file__])