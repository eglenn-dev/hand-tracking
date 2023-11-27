import mediapipe as mp
import pytest
from handTrackingModule import handTracker

methods_list = ['isThumbsUp', 'isPointingUp', 'isBird', 'isOkay', 'isFingerGun', 'isPeace', 'isLove']

def test_dataset_one():
    lmList = [[0, 354, 485], [1, 302, 456], [2, 266, 407], [3, 242, 366], [4, 218, 336], [5, 314, 335], [6, 298, 279], [7, 287, 245], [8, 278, 214], [9, 350, 330], [10, 353, 268], [11, 351, 228], [12, 349, 194], [13, 380, 341], [14, 392, 282], [15, 397, 245], [16, 398, 211], [17, 406, 361], [18, 421, 319], [19, 429, 289], [20, 435, 260]]
    test_hand = handTracker(lmList=lmList)
    assert call_set_methods(test_hand, methods_list)

def test_dataset_two():
    lmList = [[0, 179, 83], [1, 255, 105], [2, 315, 156], [3, 335, 216], [4, 333, 264], [5, 289, 203], [6, 292, 307], [7, 289, 366], [8, 284, 410], [9, 242, 210], [10, 250, 320], [11, 253, 386], [12, 255, 435], [13, 198, 211], [14, 206, 312], [15, 211, 374], [16, 216, 421], [17, 159, 208], [18, 166, 287], [19, 171, 333], [20, 175, 370]]
    test_hand = handTracker(lmList=lmList)
    assert call_set_methods(test_hand, methods_list)

def call_set_methods(obj, method_names):
    values = []

    for name in method_names:
        method = getattr(obj, name)
        try:
            if callable(method):
                if not method():
                    values.append(True)
        except Exception as e:
            values.append(False)
    if check_bool_values(values): return True
    else: return False

def check_bool_values(values):
    for val in values:
        if not val:
            return False
    return True

pytest.main(["-v", "--tb=line", "-rN", __file__])

# ================= Testing Functions ================= #

# def call_all_methods(obj):
#     # Get a list of all attributes of the object
#     all_attributes = dir(obj)

#     # Filter out only the callable methods
#     methods = [attr for attr in all_attributes if callable(getattr(obj, attr))]

#     # Call each method on the object
#     for method_name in methods:
#         method = getattr(obj, method_name)
#         try:
#             if callable(method):
#                 method()
#                 print(f"Called method '{method_name}': {result}")
#         except Exception as e:
#             print(f"Error calling method '{method_name}': {e}")