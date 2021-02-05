# Add pose library path
import os.path as osp
import sys
this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, 'pytorch-pose-hg-3d', 'src', 'lib')

if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

# Rest of the imports
from flask import Flask, Response
import cv2
from pose_model import PoseModel

app = Flask(__name__, static_url_path="/static")

@app.route('/')
def static_file():
    return app.send_static_file('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    camera = cv2.VideoCapture(0) # index specifies which camera to use
    model = PoseModel()

    # Here's how to load a single image
    # image_name = 'h36m_1214.png'
    # image = cv2.imread(image_name)
    # print(model.predict(image))

    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            frame = model.predict(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app.run()
