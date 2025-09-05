
from flask import Flask, Response, render_template, request
from transformers import pipeline
import cv2 # Added for video processing

# Load Hugging Face pipeline (sentiment analysis)
classifier = pipeline("sentiment-analysis")

app = Flask(__name__)

camera = cv2.VideoCapture(0) # Initialize camera
def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            detector=cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
            faces=detector.detectMultiScale(frame,1.1,7)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             #Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/videostream", methods=["GET", "POST"])
def videostream():
    return render_template("videostream.html")

@app.route("/livestream", methods=["GET", "POST"])
def livestream():
    return render_template("livestream.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')    

@app.route("/", methods=["GET"])
def home():
    result = None
    if request.method == "POST":
        text = request.form["user_input"]
        result = classifier(text)[0]  # Example: {'label': 'POSITIVE', 'score': 0.99}
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)