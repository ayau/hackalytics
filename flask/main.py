# Add pose library path
import os
import os.path as osp
import sys
this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, 'pytorch-pose-hg-3d', 'src', 'lib')

if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

# Rest of the imports
from flask import Flask, request, Response, session
from flask_session import Session
import cv2
from pose_model import PoseModel
from utils.debugger import show_2d, mpii_edges
import time

app = Flask(__name__, static_url_path="/static")
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

# state for uploading a video
uploads_dir = os.path.join(this_dir, 'video_uploads')

@app.route('/')
def static_file():
    return app.send_static_file('index.html')

@app.route('/upload_video', methods=['POST'])
def process_video():
    vid_obj =request.files['file']
    vid_obj.save(os.path.join(uploads_dir, vid_obj.filename))
    session['video_name'] = vid_obj.filename
    return ''

@app.route('/fetch_video_stats')
def fetch_video_stats():
    video_path = os.path.join(uploads_dir, session.get('video_name'))
    return ''

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


    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (50, 50) 
    fontScale = 1
    color = (255, 0, 0) 
    thickness = 2
    start_time = 0
    fps = "0"

    use_media_pipe = True
    use_segmentation = False
    if use_media_pipe:
        model.open_media_pipe()
    while True:
        start_time = time.time()
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        elif use_media_pipe:
            frame = model.predict_with_mediapipe(frame)
            fps = 'fps:{}'.format( int(1/(time.time()-start_time)))
            frame = cv2.putText(frame, fps , org, font, fontScale, color, thickness, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        elif use_segmentation:
            frame = model.predict_segment(frame)
            fps = 'fps:{}'.format( int(1/(time.time()-start_time)))
            frame = cv2.putText(frame, fps , org, font, fontScale, color, thickness, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        else:
            #frame = model.predict(frame)
            frame, pred, mpii_edges = model.predict(frame)
            fps = 'fps:{}'.format( int(1/(time.time()-start_time)) )
            frame = cv2.putText(frame, fps , org, font, fontScale, color, thickness, cv2.LINE_AA) 
            frame = show_2d(frame, pred, color, mpii_edges)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    model.close_media_pipe()
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app.run()
