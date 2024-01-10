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
