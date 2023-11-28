from flask import Flask, render_template, Response
import cv2
from handTrackingModule import handTracker

app = Flask(__name__)
cap = cv2.VideoCapture(0)
tracker = handTracker()

def generate_frames():
    while True:
        success, image = cap.read()
        image = tracker.handsFinder(image)
        tracker.positionFinder(image)

        if tracker.isThumbsUp():
            gesture_text = "Thumb up"
        elif tracker.isPointingUp():
            gesture_text = "Pointing up"
        elif tracker.isBird():
            gesture_text = "Bird!"
        elif tracker.isOkay():
            gesture_text = "Okay"
        elif tracker.isFingerGun():
            gesture_text = "Finger gun"
        elif tracker.isPeace():
            gesture_text = "Peace"
        elif tracker.isVictory():
            gesture_text = "Victory"
        else:
            gesture_text = "*No sign detected*"

        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'
               b'Content-Type: text/plain\r\n\r\n' + gesture_text.encode() + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
