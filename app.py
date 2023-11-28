from flask import Flask, render_template, Response
from handTrackingModule import handTracker

app = Flask(__name__)
tracker = handTracker()

def generate_gestures():
    while True:
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

        yield f"data:{gesture_text}\n\n"  # Use \n\n instead of \r\n\r\n


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/gesture_feed')
def gesture_feed():
    return Response(generate_gestures(), content_type='text/event-stream')


if __name__ == "__main__":
    app.run(debug=True)
