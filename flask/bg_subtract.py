
import cv2
import numpy as np
import time

# python3 bg_subtract.py to run it

# Settings
video_name = '../sample_videos/jump5.mp4'
only_display_largest_blob = False # only draw the largest blob to reduce noise
remove_noise = True # remove small blobs and plug some small holes
skip_frames = 1  # how many frames to process. setting to 3 will process one every 3 frames
use_cnt_model = False # Faster but less accurate

# use person detection to help narrow down search space
# To enable this, first download the yolov3.weights and yolov3.cfg from https://medium.com/@luanaebio/detecting-people-with-yolo-and-opencv-5c1f9bc6a810
# and put those in flask/
detect_person = False


def run():
    cap = cv2.VideoCapture(video_name)

    if use_cnt_model:
        fgbg = cv2.bgsegm.createBackgroundSubtractorCNT() # way faster but a bit noisier
    else:
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(backgroundRatio=0.95) # 95% of the image is the background

    frame_num = 0
    while True:
        ret, frame = cap.read()

        # Uncomment if you want to skip some frames. Maybe we can use binary search to find the launch frame
        frame_num += 1
        if frame_num % skip_frames != 0:
            continue

        start = time.time()

        # Detect person
        if detect_person:
            bounds = detectPersonBounds(frame)

        # Convert to gray scale for faster processing
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply background subtraction
        fgmask = fgbg.apply(frame)

        # Remove some noise
        # Morph open removes random small noises
        # Morph close fills in the gap in large blobs
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        if remove_noise:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # elliptical shaped kernel
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

        # Apply a mask to only focus on the person
        # If we want to do this in prod we should make the bounds a bit larger for some more leeway
        if detect_person:
            [x, y, w, h] = bounds
            mask = np.zeros(frame.shape[:2], np.uint8)
            start_rect = (int(x), int(y))
            end_rect = (int(x + w), int(y + h))
            mask = cv2.rectangle(mask, start_rect, end_rect, (255, 255, 255), -1)
            fgmask = cv2.bitwise_and(fgmask, fgmask, mask=mask)

        # Select the blob with the largest contour and hide everything else
        if only_display_largest_blob:
            inter = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))

            # Find largest contour in intermediate image
            cnts, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(cnts) > 0:
                cnt = max(cnts, key=cv2.contourArea)

                # Output
                output = np.zeros(fgmask.shape, np.uint8)
                cv2.drawContours(output, [cnt], -1, 255, cv2.FILLED)
                fgmask = cv2.bitwise_and(fgmask, output)

        print(time.time() - start)

        cv2.imshow('mask', fgmask) # show mask video
        cv2.imshow('original', frame) # show original video

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# We can first identify people in the frame, then apply background segmentation within those bounds to avoid noise
# https://medium.com/@luanaebio/detecting-people-with-yolo-and-opencv-5c1f9bc6a810
# channels = 1 in yolov3.cfg for grayscale. If not using grayscale, change this back to 3
# read pre-trained model and config file
if detect_person:
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

def detectPersonBounds(image):

    # create input blob
    scale = 0.00392
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)

    # run inference through the network
    # and gather predictions from output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    # Find Bounding Box
    box = None  # TODO handle multiple people in the image (find the largest one)
    confidence_threshold = 0.1
    person_class_id = 0
    width = image.shape[1]
    height = image.shape[0]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold and class_id == person_class_id:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                box = [x, y, w, h]

                # If we want to draw the box
                # cv2.rectangle(image, (round(box[0]), round(box[1])), (round(box[0] + box[2]), round(box[1] + box[3])), (255, 0, 0), 2)
                # cv2.putText(image, 'Person', (round(box[0]) - 10, round(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return box


run()