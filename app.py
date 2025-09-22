from flask import Flask, render_template, Response, request, redirect
import cv2
import threading
import numpy as np
import imutils
from utils.attention import AttentionTracker
from utils.authenticity import check_face_authenticity

app = Flask(__name__)
tracker = AttentionTracker()
calibrated = False

# ---------------- Threaded camera ----------------
class VideoCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.grabbed, self.frame = self.capture.read()
        self.lock = threading.Lock()
        t = threading.Thread(target=self.update, daemon=True)
        t.start()

    def update(self):
        while True:
            grabbed, frame = self.capture.read()
            if grabbed:
                frame = imutils.resize(frame, width=480)
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy()

camera = VideoCamera(0)

# ---------------- Frame generator ----------------
def gen_frames():
    global calibrated
    frame_count = 0
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        frame_count += 1
        if frame_count % 3 != 0:  # process every 3rd frame
            continue

        # Calibration overlay
        if not calibrated:
            cv2.putText(frame, "Calibration in Progress - Look Straight!", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            tracker.calibrate(frame, samples=30)
            calibrated = True

        attention, _ = tracker.get_attention(frame)
        authenticity = check_face_authenticity(frame)

        cv2.putText(frame, f"Attention: {attention}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        cv2.putText(frame, f"Authenticity: {authenticity}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ---------------- Flask routes ----------------
@app.route('/', methods=['GET', 'POST'])
def index():
    authenticity = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        authenticity = check_face_authenticity(img)
    return render_template('index.html', authenticity=authenticity)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------- Run app ----------------
if __name__ == "__main__":
    # CPU-friendly, disables auto-reloader to prevent "Restarting with stat"
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False, threaded=True) 
  

