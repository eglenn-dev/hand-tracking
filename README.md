# Hand Tracker Class Documentation

## Overview

The `handTracker` class is a Python implementation for hand tracking using the MediaPipe library. This class provides a convenient interface for detecting and tracking hands in images or video frames. It also includes functions to identify hand gestures such as thumbs up, pointing, bird, okay, finger gun, peace sign, victory, and the letter 'O'. The class utilizes the MediaPipe Hands module for accurate hand landmark estimation.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Methods](#methods)
  - [Constructor](#constructor)
  - [handsFinder](#handsFinder)
  - [positionFinder](#positionFinder)
  - [isThumbsUp](#isThumbsUp)
  - [isPointingUp](#isPointingUp)
  - [isBird](#isBird)
  - [isOkay](#isOkay)
  - [isFingerGun](#isFingerGun)
  - [isPeace](#isPeace)
  - [isVictory](#isVictory)
  - [isO](#isO)
  - [isAbove](#isAbove)
  - [handOrientation](#handOrientation)
  - [handDirection](#handDirection)
  - [calculate_average_position](#calculate_average_position)
  - [calculate_distance](#calculate_distance)
  - [calculate_triangle_angles](#calculate_triangle_angles)

## Installation<a name="installation"></a>

To use the `handTracker` class, you need to have the following libraries installed:

```bash
pip install opencv-python mediapipe
```

## Usage<a name="usage"></a>

Here's a basic example of how to use the handTracker class:

```{python}
import cv2
import mediapipe as mp
from hand_tracker import handTracker

# Initialize hand tracker
tracker = handTracker()

# Read an image or video frame
image = cv2.imread("hand.jpg")

# Find hands in the image and draw landmarks
image_with_hands = tracker.handsFinder(image, draw=True)

# Get hand positions
hand_positions = tracker.positionFinder(image)

# Display the results
cv2.imshow("Hand Tracker", image_with_hands)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

This would work on a still image, so ideally it would be run in some kind of loop with an exit key. See [here](./app.py).

## Methods<a name="methods"></a>

Please note that for many of these methods there are defaukt vlaues of none. This is for code resuability. Want's to run a check on a preset hand position, then they can input a value. Otherwise if the defult none is used, the methods will use the list from the object.

### Constructor<a name="constructor"></a>

```{python}
def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5, lmList=[]):
    """
    Initializes the handTracker object.

    Parameters:
    - mode (bool): Whether to run in tracking mode.
    - maxHands (int): Maximum number of hands to detect.
    - detectionCon (float): Minimum detection confidence.
    - modelComplexity (int): Model complexity (1 or 2).
    - trackCon (float): Minimum tracking confidence.
    - lmList (list): List to store landmarks.
    """
```

### handsFinder<a name="handsFinder"></a>

```{python}
def handsFinder(self, image, draw=True):
    """
    Detects hands in the given image and draws landmarks if draw is True.

    Parameters:
    - image: Image to process.
    - draw (bool): Whether to draw landmarks.

    Returns:
    - image: Image with landmarks drawn.
    """
```

### positionFinder<a name="positionFinder"></a>

```{python}
def positionFinder(self, image, handNo=0):
    """
    Finds the positions of hand landmarks in the given image. Writes to object list.

    Parameters:
    - image: Image to process.
    - handNo (int): Index of the hand to track.

    Returns:
    None
    """
```

### isThumbsUp<a name="isThumbsUp"></a>

```{python}
def isThumbsUp(self, lmList=None):
    """
    Checks if the hand gesture represents a thumbs-up.

    Parameters:
    - lmList (list): List of hand landmarks.

    Returns:
    - bool: True if the gesture is thumbs-up, False otherwise.
    """
```

### isPointingUp<a name="isPointingUp"></a>

```{python}
def isPointingUp(self, lmList=None):
    """
    Checks if the hand gesture represents pointing upwards.

    Parameters:
    - lmList (list): List of hand landmarks.

    Returns:
    - bool: True if the gesture is pointing up, False otherwise.
    """
```

### isBird<a name="isBird"></a>

```{python}
def isBird(self, lmList=None):
    """
    Checks if the hand gesture represents a 'bird'.

    Parameters:
    - lmList (list): List of hand landmarks.

    Returns:
    - bool: True if the gesture is a bird, False otherwise.
    """
```

### isOkay<a name="isOkay"></a>

```{python}
def isOkay(self, lmList=None):
    """
    Checks if the hand gesture represents an 'Okay' sign.

    Parameters:
    - lmList (list): List of hand landmarks.

    Returns:
    - bool: True if the gesture is 'Okay', False otherwise.
    """
```

### isFingerGun<a name="isFingerGun"></a>

```{python}
def isFingerGun(self, lmList=None):
    """
    Checks if the hand gesture represents a finger gun.

    Parameters:
    - lmList (list): List of hand landmarks.

    Returns:
    - bool: True if the gesture is a finger gun, False otherwise.
    """
```

### isPeace<a name="isPeace"></a>

```{python}
def isPeace(self, lmList=None):
    """
    Checks if the hand gesture represents a peace sign.

    Parameters:
    - lmList (list): List of hand landmarks.

    Returns:
    - bool: True if the gesture is a peace sign, False otherwise.
    """
```

### isVictory<a name="isVictory"></a>

```{python}
def isVictory(self, lmList=None):
    """
    Checks if the hand gesture represents a victory sign.

    Parameters:
    - lmList (list): List of hand landmarks.

    Returns:
    - bool: True if the gesture is a victory sign, False otherwise.
    """
```

### isO<a name="isO"></a>

```{python}
def isO(self, lmList=None):
    """
    Checks if the hand gesture represents the letter 'O'.

    Parameters:
    - lmList (list): List of hand landmarks.

    Returns:
    - bool: True if the gesture is the letter 'O', False otherwise.
    """
```

### isAbove<a name="isAbove"></a>

```{python}
def isAbove(self, target, landmarks, lmList=None):
    """
    Checks if the specified landmarks are above the target landmark.

    Parameters:
    - target (list): Target landmark [id, x, y].
    - landmarks (list): List of landmark ids to compare.
    - lmList (list): List of hand landmarks.

    Returns:
    - bool: True if all specified landmarks are above the target, False otherwise.
    """
```

### handOrientation<a name="handOrientation"></a>

```{python}
def handOrientation(self, orientation, lmList=None):
    """
    Checks if the hand orientation is as specified (up or down).

    Parameters:
    - orientation (str): Desired orientation ('up' or 'down').
    - lmList (list): List of hand landmarks.

    Returns:
    - bool: True if the hand orientation matches the specified orientation, False otherwise.
    """
```

### handDirection<a name="handDirection"></a>

```{python}
def handDirection(self, direction, lmList=None):
    """
    Checks if the hand direction is specified (left or right).

    Parameters:
    - direction (str): Desired direction ('left' or 'right').
    - lmList (list): List of hand landmarks.

    Returns:
    - bool: True if the hand direction matches the specified direction, False otherwise.
    """
```

### calculate_average_position<a name="calculate_average_position"></a>

```{python}
def calculate_average_position(self, landmarks, lmList=None):
    """
    Calculates the average position of specified landmarks.

    Parameters:
    - landmarks (list): List of landmark ids.
    - lmList (list): List of hand landmarks.

    Returns:
    - list: [0, average_x, average_y].
    """
```

### calculate_distance<a name="calculate_distance"></a>

```{python}
@staticmethod
def calculate_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points.

    Parameters:
    - point1 (list): First point [id, x, y].
    - point2 (list): Second point [id, x, y].

    Returns:
    - float: Euclidean distance between the two points.
    """
```

### calculate_triangle_angles<a name="calculate_triangle_angles"></a>

```{python}
@staticmethod
def calculate_triangle_angles(a, b, c):
    """
    Calculates the angles of a triangle given its side lengths.

    Parameters:
    - a (float): Length of side a.
    - b (float): Length of side b.
    - c (float): Length of side c.

    Returns:
    - tuple: (angleA, angleB, angleC).
    """
```
