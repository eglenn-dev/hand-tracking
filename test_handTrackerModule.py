import mediapipe as mp
import pytest
from pytest import approx
from handTrackingModule import handTracker

hand = handTracker()
methods_list = ['isThumbsUp', 'isPointingUp', 'isBird', 'isOkay', 'isFingerGun', 'isPeace', 'isVictory', 'isO']

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
    test_hand = handTracker(lmList=lmList_isAbove)
    assert test_hand.isAbove(lmList_isAbove[20], [8, 12, 16])
    assert test_hand.isAbove(lmList_isAbove[20], [5, 9, 13])
    assert not test_hand.isAbove(lmList_isAbove[0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    assert not test_hand.isAbove(lmList_isAbove[8], [18, 19, 20])

def test_pointingUp():
    lmList_pointing_up = [[0, 217, 494], [1, 260, 478], [2, 299, 449], [3, 320, 417], [4, 300, 398], [5, 272, 352], [6, 280, 291], [7, 285, 253], [8, 285, 221], [9, 236, 350], [10, 280, 332], [11, 288, 390], [12, 276, 415], [13, 206, 361], [14, 253, 268], [15, 257, 416], [16, 241, 430], [17, 178, 382], [18, 221, 401], [19, 230, 432], [20, 217, 439]]
    test_hand = handTracker(lmList=lmList_pointing_up)
    expected = [False] * len(methods_list)
    expected[1] = True
    for i in range(len(methods_list)):
        method = getattr(test_hand, methods_list[i])
        if callable(method):
            result = method()
            assert result == expected[i], f'Failed on {methods_list[i]}'

def test_isFingerGun():
    lmList_finger_gun = [[0, 148, 386], [1, 155, 309], [2, 192, 246], [3, 233, 198], [4, 244, 155], [5, 264, 240], [6, 363, 227], [7, 420, 230], [8, 465, 236], [9, 281, 277], [10, 347, 309], [11, 315, 322], [12, 281, 319], [13, 287, 326], [14, 334, 352], [15, 304, 362], [16, 274, 360], [17, 291, 374], [18, 318, 389], [19, 289, 396], [20, 264, 392]]
    test_hand = handTracker(lmList=lmList_finger_gun)
    expected = [False] * len(methods_list)
    expected[4] = True
    for i in range(len(methods_list)):
        method = getattr(test_hand, methods_list[i])
        if callable(method):
            result = method()
            assert result == expected[i], f'Failed on {methods_list[i]}'
    
def test_isOkay():
    lmList_okay = [[0, 258, 406], [1, 310, 371], [2, 351, 326], [3, 372, 287], [4, 382, 246], [5, 286, 247], [6, 317, 202], [7, 348, 206], [8, 372, 226], [9, 254, 234], [10, 250, 160], [11, 268, 115], [12, 286, 80], [13, 225, 239], [14, 204, 170], [15, 211, 125], [16, 225, 87], [17, 200, 257], [18, 164, 209], [19, 151, 171], [20, 147, 132]]
    test_hand = handTracker(lmList=lmList_okay)
    expected = [False] * len(methods_list)
    expected[3] = True
    for i in range(len(methods_list)):
        method = getattr(test_hand, methods_list[i])
        if callable(method):
            result = method()
            assert result == expected[i], f'Failed on {methods_list[i]}'

def test_isBird():
    lmList_bird = [[0, 204, 451], [1, 159, 399], [2, 139, 343], [3, 166, 300], [4, 205, 293], [5, 184, 287], [6, 215, 251], [7, 221, 274], [8, 219, 298], [9, 227, 292], [10, 265, 218], [11, 286, 172], [12, 303, 143], [13, 265, 310], [14, 296, 255], [15, 286, 255], [16, 274, 298], [17, 298, 342], [18, 315, 295], [19, 301, 311], [20, 291, 334]]
    test_hand = handTracker(lmList=lmList_bird)
    expected = [False] * len(methods_list)
    expected[2] = True
    for i in range(len(methods_list)):
        method = getattr(test_hand, methods_list[i])
        if callable(method):
            result = method()
            assert result == expected[i], f'Failed on {methods_list[i]}'

def test_isThumbUp():
    lmList_thumb_up = [[0, 207, 379], [1, 220, 305], [2, 258, 239], [3, 310, 195], [4, 341, 159], [5, 327, 227], [6, 403, 261], [7, 385, 283], [8, 357, 285], [9, 342, 271], [10, 412, 307], [11, 386, 325], [12, 356, 322], [13, 348, 321], [14, 407, 353], [15, 378, 365], [16, 349, 360], [17, 348, 372], [18, 391, 393], [19, 362, 401], [20, 334, 395]]
    test_hand = handTracker(lmList=lmList_thumb_up)
    expected = [False] * len(methods_list)
    expected[0] = True
    for i in range(len(methods_list)):
        method = getattr(test_hand, methods_list[i])
        if callable(method):
            result = method()
            assert result == expected[i], f'Failed on {methods_list[i]}'

def test_isHandUp():
    lmList_hand_up = [[0, 225, 417], [1, 297, 390], [2, 349, 336], [3, 379, 286], [4, 408, 248], [5, 290, 234], [6, 319, 162], [7, 337, 115], [8, 350, 75], [9, 243, 224], [10, 248, 139], [11, 252, 85], [12, 253, 39], [13, 201, 235], [14, 189, 158], [15, 183, 106], [16, 179, 61], [17, 163, 262], [18, 130, 209], [19, 110, 171], [20, 96, 134]]
    test_hand = handTracker(lmList=lmList_hand_up)
    expected = [False] * len(methods_list)
    assert test_hand.handOrientation("up")
    assert not test_hand.handOrientation("down")
    for i in range(len(methods_list)):
        method = getattr(test_hand, methods_list[i])
        if callable(method):
            result = method()
            assert result == expected[i], f'Failed on {methods_list[i]}'

def test_isHandDown():
    lmList_hand_down = [[0, 233, -11], [1, 317, 21], [2, 386, 82], [3, 428, 140], [4, 475, 167], [5, 375, 137], [6, 402, 262], [7, 413, 340], [8, 419, 402], [9, 321, 138], [10, 326, 281], [11, 329, 366], [12, 330, 428], [13, 263, 135], [14, 244, 272], [15, 230, 350], [16, 221, 412], [17, 206, 126], [18, 173, 225], [19, 150, 276], [20, 132, 322]]
    test_hand = handTracker(lmList=lmList_hand_down)
    expected = [False] * len(methods_list)
    assert not test_hand.handOrientation("up")
    assert test_hand.handOrientation("down")
    for i in range(len(methods_list)):
        method = getattr(test_hand, methods_list[i])
        if callable(method):
            result = method()
            assert result == expected[i], f'Failed on {methods_list[i]}'

def test_isPeace():
    lmList_peace = [[0, 229, 407], [1, 272, 393], [2, 299, 359], [3, 292, 328], [4, 268, 306], [5, 288, 282], [6, 311, 234], [7, 322, 204], [8, 329, 177], [9, 257, 278], [10, 252, 216], [11, 251, 180], [12, 247, 149], [13, 227, 291], [14, 237, 256], [15, 252, 295], [16, 256, 322], [17, 201, 314], [18, 220, 295], [19, 235, 321], [20, 239, 342]]
    test_hand = handTracker(lmList=lmList_peace)
    expected = [False] * len(methods_list)
    expected[5] = True
    for i in range(len(methods_list)):
        method = getattr(test_hand, methods_list[i])
        if callable(method):
            result = method()
            assert result == expected[i], f'Failed on {methods_list[i]}'

def test_isVictory():
    lmList_love = [[0, 424, 435], [1, 368, 405], [2, 336, 352], [3, 331, 300], [4, 329, 256], [5, 369, 286], [6, 347, 226], [7, 337, 193], [8, 330, 160], [9, 402, 277], [10, 380, 215], [11, 367, 178], [12, 354, 143], [13, 435, 284], [14, 446, 224], [15, 457, 182], [16, 463, 145], [17, 466, 303], [18, 475, 255], [19, 480, 223], [20, 483, 292]]
    test_hand = handTracker(lmList=lmList_love)
    expected = [False] * len(methods_list)
    expected[6] = True
    for i in range(len(methods_list)):
        method = getattr(test_hand, methods_list[i])
        if callable(method):
            result = method()
            assert result == expected[i], f'Failed on {methods_list[i]}'

def test_isO():
    lmList_O = [[0, 1215, 470], [1, 1130, 426], [2, 1070, 366], [3, 1028, 320], [4, 1008, 273], [5, 1135, 275], [6, 1105, 205], [7, 1057, 212], [8, 1029, 237], [9, 1157, 267], [10, 1118, 179], [11, 1056, 194], [12, 1025, 231], [13, 1180, 267], [14, 1141, 178], [15, 1074, 196], [16, 1039, 235], [17, 1196, 274], [18, 1148, 204], [19, 1092, 210], [20, 1059, 239]]
    test_hand = handTracker(lmList=lmList_O)
    expected = [False] * len(methods_list)
    expected[7] = True
    for i in range(len(methods_list)):
        method = getattr(test_hand, methods_list[i])
        if callable(method):
            result = method()
            assert result == expected[i], f'Failed on {methods_list[i]}'

def test_isHandLeft():
    lmList_hand_left = [[0, 186, 362], [1, 202, 298], [2, 232, 260], [3, 267, 235], [4, 296, 210], [5, 305, 263], [6, 391, 246], [7, 438, 244], [8, 474, 246], [9, 320, 298], [10, 418, 288], [11, 475, 286], [12, 514, 287], [13, 323, 338], [14, 417, 341], [15, 471, 347], [16, 510, 352], [17, 314, 377], [18, 386, 390], [19, 426, 400], [20, 458, 408]]
    test_hand = handTracker(lmList=lmList_hand_left)
    assert test_hand.handDirection("left")
    assert not test_hand.handDirection("right")

def test_isHandRight():
    lmList_hand_right = [[0, 492, 342], [1, 480, 261], [2, 445, 203], [3, 406, 157], [4, 392, 114], [5, 369, 202], [6, 279, 184], [7, 224, 180], [8, 178, 178], [9, 352, 240], [10, 247, 233], [11, 185, 229], [12, 137, 227], [13, 345, 285], [14, 246, 285], [15, 188, 288], [16, 142, 290], [17, 348, 331], [18, 273, 341], [19, 232, 351], [20, 196, 356]]
    test_hand = handTracker(lmList=lmList_hand_right)
    assert test_hand.handDirection("right")
    assert not test_hand.handDirection("left")

def check_average_calculator(lmList, landmarks, expected):
    test_hand = handTracker(lmList=lmList)
    items = test_hand.calculate_average_position(landmarks)
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