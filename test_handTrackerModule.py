import cv2
import mediapipe as mp
import pytest
from pytest import approx
from handTrackingModule import handTracker

hand = handTracker()

@pytest.fixture
def hand_tracker_instance():
    return handTracker()

def test_array():
    lmList1 = [[0, 354, 485], [1, 302, 456], [2, 266, 407], [3, 242, 366], [4, 218, 336], [5, 314, 335], [6, 298, 279], [7, 287, 245], [8, 278, 214], [9, 350, 330], [10, 353, 268], [11, 351, 228], [12, 349, 194], [13, 380, 341], [14, 392, 282], [15, 397, 245], [16, 398, 211], [17, 406, 361], [18, 421, 319], [19, 429, 289], [20, 435, 260]]
    lmList2 = [[0, 179, 83], [1, 255, 105], [2, 315, 156], [3, 335, 216], [4, 333, 264], [5, 289, 203], [6, 292, 307], [7, 289, 366], [8, 284, 410], [9, 242, 210], [10, 250, 320], [11, 253, 386], [12, 255, 435], [13, 198, 211], [14, 206, 312], [15, 211, 374], [16, 216, 421], [17, 159, 208], [18, 166, 287], [19, 171, 333], [20, 175, 370]]
    assert check_array_for_values(lmList1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    assert check_array_for_values(lmList1, [19, 18, 14, 15, 2, 0])
    assert not check_array_for_values(lmList1, [19, 18, 14, 15, 2, 21])
    assert check_array_for_values(lmList2, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    assert check_array_for_values(lmList2, [19, 18, 14, 15, 2, 0])
    assert not check_array_for_values(lmList2, [19, 18, 14, 15, 2, 21])

def test_calculate_average_position():
    lmList = [[0, 354, 485], [1, 302, 456], [2, 266, 407], [3, 242, 366], [4, 218, 336], [5, 314, 335], [6, 298, 279], [7, 287, 245], [8, 278, 214], [9, 350, 330], [10, 353, 268], [11, 351, 228], [12, 349, 194], [13, 380, 341], [14, 392, 282], [15, 397, 245], [16, 398, 211], [17, 406, 361], [18, 421, 319], [19, 429, 289], [20, 435, 260]]
    check_average_calculator(lmList, [5, 9, 13, 17], [0, 362.5, 341.75])
    check_average_calculator(lmList, [0, 4, 6, 7, 19, 20], [0, 336.84, 315.67])

def test_isAbove():
    lmList_isAbove = [[0, 401, 392], [1, 336, 365], [2, 284, 309], [3, 266, 253], [4, 303, 238], [5, 322, 235], [6, 309, 190], [7, 314, 234], [8, 323, 269], [9, 362, 225], [10, 351, 174], [11, 350, 231], [12, 354, 273], [13, 403, 222], [14, 400, 160], [15, 387, 206], [16, 379, 247], [17, 442, 228], [18, 450, 172], [19, 451, 139], [20, 449, 110]]
    assert handTracker().isAbove(lmList_isAbove[20], lmList_isAbove, [8, 12, 16])
    assert handTracker().isAbove(lmList_isAbove[20], lmList_isAbove, [5, 9, 13])
    assert not handTracker().isAbove(lmList_isAbove[0], lmList_isAbove, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    assert not handTracker().isAbove(lmList_isAbove[8], lmList_isAbove, [18, 19, 20])

def test_pointingUp():
    lmList_pointing_up = [[0, 217, 494], [1, 260, 478], [2, 299, 449], [3, 320, 417], [4, 300, 398], [5, 272, 352], [6, 280, 291], [7, 285, 253], [8, 285, 221], [9, 236, 350], [10, 280, 332], [11, 288, 390], [12, 276, 415], [13, 206, 361], [14, 253, 268], [15, 257, 416], [16, 241, 430], [17, 178, 382], [18, 221, 401], [19, 230, 432], [20, 217, 439]]
    assert handTracker().isPointingUp(lmList_pointing_up)
    assert not handTracker().isThumbsUp(lmList_pointing_up)
    assert not handTracker().isBird(lmList_pointing_up)
    assert not handTracker().isFingerGun(lmList_pointing_up)
    assert not handTracker().isOkay(lmList_pointing_up)
    assert not handTracker().isPeace(lmList_pointing_up)
    assert handTracker().handOrientation(lmList_pointing_up, "up")
    assert not handTracker().handOrientation(lmList_pointing_up, "down")

def test_isFingerGun():
    lmList_finger_gun = [[0, 148, 386], [1, 155, 309], [2, 192, 246], [3, 233, 198], [4, 244, 155], [5, 264, 240], [6, 363, 227], [7, 420, 230], [8, 465, 236], [9, 281, 277], [10, 347, 309], [11, 315, 322], [12, 281, 319], [13, 287, 326], [14, 334, 352], [15, 304, 362], [16, 274, 360], [17, 291, 374], [18, 318, 389], [19, 289, 396], [20, 264, 392]]
    assert handTracker().isFingerGun(lmList_finger_gun)
    assert not handTracker().isPointingUp(lmList_finger_gun)
    assert not handTracker().isThumbsUp(lmList_finger_gun)
    assert not handTracker().isBird(lmList_finger_gun)
    assert not handTracker().isOkay(lmList_finger_gun)
    assert not handTracker().isPeace(lmList_finger_gun)
    assert handTracker().handDirection(lmList_finger_gun, "left")
    assert not handTracker().handDirection(lmList_finger_gun, "right")

def test_isOkay():
    lmList_okay = [[0, 258, 406], [1, 310, 371], [2, 351, 326], [3, 372, 287], [4, 382, 246], [5, 286, 247], [6, 317, 202], [7, 348, 206], [8, 372, 226], [9, 254, 234], [10, 250, 160], [11, 268, 115], [12, 286, 80], [13, 225, 239], [14, 204, 170], [15, 211, 125], [16, 225, 87], [17, 200, 257], [18, 164, 209], [19, 151, 171], [20, 147, 132]]
    assert handTracker().isOkay(lmList_okay)
    assert not handTracker().isFingerGun(lmList_okay)
    assert not handTracker().isPointingUp(lmList_okay)
    assert not handTracker().isThumbsUp(lmList_okay)
    assert not handTracker().isBird(lmList_okay)
    assert not handTracker().isPeace(lmList_okay)
    assert not handTracker().handDirection(lmList_okay, "left")
    assert handTracker().handDirection(lmList_okay, "right")

def test_isBird():
    lmList_bird = [[0, 204, 451], [1, 159, 399], [2, 139, 343], [3, 166, 300], [4, 205, 293], [5, 184, 287], [6, 215, 251], [7, 221, 274], [8, 219, 298], [9, 227, 292], [10, 265, 218], [11, 286, 172], [12, 303, 143], [13, 265, 310], [14, 296, 255], [15, 286, 255], [16, 274, 298], [17, 298, 342], [18, 315, 295], [19, 301, 311], [20, 291, 334]]
    assert handTracker().isBird(lmList_bird)
    assert not handTracker().isOkay(lmList_bird)
    assert not handTracker().isFingerGun(lmList_bird)
    assert not handTracker().isPointingUp(lmList_bird)
    assert not handTracker().isThumbsUp(lmList_bird)
    assert not handTracker().isPeace(lmList_bird)
    assert handTracker().handOrientation(lmList_bird, "up")
    assert not handTracker().handOrientation(lmList_bird, "down")

def test_isThumbUp():
    lmList_thumb_up = [[0, 207, 379], [1, 220, 305], [2, 258, 239], [3, 310, 195], [4, 341, 159], [5, 327, 227], [6, 403, 261], [7, 385, 283], [8, 357, 285], [9, 342, 271], [10, 412, 307], [11, 386, 325], [12, 356, 322], [13, 348, 321], [14, 407, 353], [15, 378, 365], [16, 349, 360], [17, 348, 372], [18, 391, 393], [19, 362, 401], [20, 334, 395]]
    assert handTracker().isThumbsUp(lmList_thumb_up)
    assert not handTracker().isBird(lmList_thumb_up)
    assert not handTracker().isOkay(lmList_thumb_up)
    assert not handTracker().isFingerGun(lmList_thumb_up)
    assert not handTracker().isPointingUp(lmList_thumb_up)
    assert not handTracker().isPeace(lmList_thumb_up)
    assert handTracker().handDirection(lmList_thumb_up, "left")
    assert not handTracker().handDirection(lmList_thumb_up, "right")

def test_isHandUp():
    lmList_hand_up = [[0, 225, 417], [1, 297, 390], [2, 349, 336], [3, 379, 286], [4, 408, 248], [5, 290, 234], [6, 319, 162], [7, 337, 115], [8, 350, 75], [9, 243, 224], [10, 248, 139], [11, 252, 85], [12, 253, 39], [13, 201, 235], [14, 189, 158], [15, 183, 106], [16, 179, 61], [17, 163, 262], [18, 130, 209], [19, 110, 171], [20, 96, 134]]
    assert not handTracker().isThumbsUp(lmList_hand_up)
    assert not handTracker().isBird(lmList_hand_up)
    assert not handTracker().isOkay(lmList_hand_up)
    assert not handTracker().isFingerGun(lmList_hand_up)
    assert not handTracker().isPointingUp(lmList_hand_up)
    assert not handTracker().isPeace(lmList_hand_up)
    assert handTracker().handOrientation(lmList_hand_up, "up")
    assert not handTracker().handOrientation(lmList_hand_up, "down")

def test_isHandDown():
    lmList_hand_down = [[0, 233, -11], [1, 317, 21], [2, 386, 82], [3, 428, 140], [4, 475, 167], [5, 375, 137], [6, 402, 262], [7, 413, 340], [8, 419, 402], [9, 321, 138], [10, 326, 281], [11, 329, 366], [12, 330, 428], [13, 263, 135], [14, 244, 272], [15, 230, 350], [16, 221, 412], [17, 206, 126], [18, 173, 225], [19, 150, 276], [20, 132, 322]]
    assert not handTracker().isThumbsUp(lmList_hand_down)
    assert not handTracker().isBird(lmList_hand_down)
    assert not handTracker().isOkay(lmList_hand_down)
    assert not handTracker().isFingerGun(lmList_hand_down)
    assert not handTracker().isPointingUp(lmList_hand_down)
    assert not handTracker().isPeace(lmList_hand_down)
    assert not handTracker().handOrientation(lmList_hand_down, "up")
    assert handTracker().handOrientation(lmList_hand_down, "down")

def test_isPeace():
    lmList_peace = [[0, 229, 407], [1, 272, 393], [2, 299, 359], [3, 292, 328], [4, 268, 306], [5, 288, 282], [6, 311, 234], [7, 322, 204], [8, 329, 177], [9, 257, 278], [10, 252, 216], [11, 251, 180], [12, 247, 149], [13, 227, 291], [14, 237, 256], [15, 252, 295], [16, 256, 322], [17, 201, 314], [18, 220, 295], [19, 235, 321], [20, 239, 342]]
    assert handTracker().isPeace(lmList_peace)
    assert not handTracker().isThumbsUp(lmList_peace)
    assert not handTracker().isBird(lmList_peace)
    assert not handTracker().isOkay(lmList_peace)
    assert not handTracker().isFingerGun(lmList_peace)
    assert not handTracker().isPointingUp(lmList_peace)
    assert handTracker().handOrientation(lmList_peace, "up")
    assert not handTracker().handOrientation(lmList_peace, "down")

def test_isHandLeft():
    lmList_hand_left = [[0, 186, 362], [1, 202, 298], [2, 232, 260], [3, 267, 235], [4, 296, 210], [5, 305, 263], [6, 391, 246], [7, 438, 244], [8, 474, 246], [9, 320, 298], [10, 418, 288], [11, 475, 286], [12, 514, 287], [13, 323, 338], [14, 417, 341], [15, 471, 347], [16, 510, 352], [17, 314, 377], [18, 386, 390], [19, 426, 400], [20, 458, 408]]
    assert handTracker().handDirection(lmList_hand_left, "left")
    assert not handTracker().handDirection(lmList_hand_left, "right")

def test_isHandRight():
    lmList_hand_right = [[0, 492, 342], [1, 480, 261], [2, 445, 203], [3, 406, 157], [4, 392, 114], [5, 369, 202], [6, 279, 184], [7, 224, 180], [8, 178, 178], [9, 352, 240], [10, 247, 233], [11, 185, 229], [12, 137, 227], [13, 345, 285], [14, 246, 285], [15, 188, 288], [16, 142, 290], [17, 348, 331], [18, 273, 341], [19, 232, 351], [20, 196, 356]]
    assert handTracker().handDirection(lmList_hand_right, "right")
    assert not handTracker().handDirection(lmList_hand_right, "left")

def check_average_calculator(lmList, landmarks, expected):
    items = handTracker.calculate_average_position(lmList, landmarks)
    for i in range(len(items)):
        assert items[i] == approx(expected[i])

def check_array_for_values(lmList, landmarks):
    values = []
    try:
        for item in landmarks:
            if item in lmList[item]: values.append(True)
            else: values.append(False)
        for val in values:
            if not val:
                return False
        return True
    except: return False


pytest.main(["-v", "--tb=line", "-rN", __file__])