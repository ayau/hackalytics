
import cv2
import numpy as np
import time
import os
import sys
import pandas as pd
# python3 bg_subtract.py to run it

# Settings

only_display_largest_blob = False # only draw the largest blob to reduce noise
remove_noise = True # remove small blobs and plug some small holes
skip_frames = 1  # how many frames to process. setting to 3 will process one every 3 frames
use_cnt_model = False # Faster but less accurate
crop_height = 15 #the height of the crop up from feet
foot_width = 15
video_width = 960
jpg_quality = 25

# use person detection to help narrow down search space
# To enable this, first download the yolov3.weights and yolov3.cfg from https://medium.com/@luanaebio/detecting-people-with-yolo-and-opencv-5c1f9bc6a810
# and put those in flask/
detect_person = False # ivan: i tried it with various sizes of the yolov model - didn't seem to improve much


def render_assets():
    for i in range(1,7):
        video_nbmr = str(i)
        os.makedirs('../media/'+video_nbmr+'/foot_placement', exist_ok=True)
        os.makedirs('../media/'+video_nbmr+'/video_frames', exist_ok=True)    
        os.makedirs('../media/'+video_nbmr+'/stats', exist_ok=True)
        os.makedirs('../media/'+video_nbmr+'/blob_placement', exist_ok=True)
        lowest_point, first_frame, last_frame = run(video_nbmr=str(i)) #first pass
        _, _, _ = run(lowest_point, first_frame, last_frame, True, video_nbmr=str(i))
        

def run(lowest_point=None, first_frame=None, last_frame=None, second_pass=False, video_nbmr=5):
    folder = '../sample_videos/'
    video_name = folder+'jump'+video_nbmr+'_small.mp4'
    cap = cv2.VideoCapture(video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    #print(fps)
    #sys.exit()

    if use_cnt_model:
        fgbg = cv2.bgsegm.createBackgroundSubtractorCNT() # way faster but a bit noisier
    else:
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    frame_num = 0
    data = []
    last_left, last_right, last_bottom, last_frame_num = (video_width, video_width, 0, 0) 
    while True:

        frame_num += 1
        _, frame = cap.read()
        if(frame is None): break

        # Uncomment if you want to skip some frames. Maybe we can use binary search to find the launch frame
        if frame_num % skip_frames != 0:
            continue

        #second pass, slice video
        if(second_pass):
            if(frame_num==30):
                stats_frame = frame.copy()


            if(first_frame):
                if(frame_num < first_frame):
                    continue

            if(last_frame):
                if(frame_num > last_frame):
                    break
            
            frame_path = "../media/"+video_nbmr+"/video_frames/"+str(frame_num)+".jpg"
            cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])



        start = time.time()

        # Detect person
        if detect_person:
            bounds = detectPersonBounds(frame)

        # Convert to gray scale for faster processing
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # applying blur reduced the noise and avoid overfitting on the background
        frame = cv2.GaussianBlur(frame, (11, 11), 0) # must be an odd number
        # frame = cv2.medianBlur(frame, 5)

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
        if detect_person and bounds: #sometimes there were no bounds and this crashed
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

        # looks for non zero items on the mask
        positions_y, positions_x = np.nonzero(fgmask)




        if(len(positions_x)>0):

            #tracks first and last frame with something on the mask
            if(first_frame is None):
                first_frame = frame_num
                lowest_point = 0

            if not second_pass:
                last_frame= frame_num

            if second_pass:
                #use positions before trimming for blob visualization only
                top = positions_y.min()
                bottom = positions_y.max()
                left = positions_x.min()
                right = positions_x.max()

                #save blob mask for static site
                blobmask = cv2.rectangle(cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB), (left, top), (right, bottom), (0,255,0), 1)
                frame_path = "../media/"+video_nbmr+"/blob_placement/"+str(frame_num)+".png"
                cv2.imwrite(frame_path, blobmask, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])

                #only look at bottom slice
                positions_x = np.delete(positions_x, np.where(positions_y<lowest_point-crop_height))
                positions_y = np.delete(positions_y, np.where(positions_y<lowest_point-crop_height))
                

            if not (len(positions_x)>0): continue
            top = positions_y.min()
            bottom = positions_y.max()
            left = positions_x.min()
            right = positions_x.max()

            #foot placement frame store
            frame_path = "../media/"+video_nbmr+"/foot_placement/"+str(frame_num)+".png"
            cv2.imwrite(frame_path, fgmask, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            if (bottom > lowest_point): lowest_point = bottom


            fgmask = cv2.rectangle(cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
                , (left, lowest_point - crop_height), (right, lowest_point), (0,255,0), 1)




            if(second_pass):
                crop_fgmask = fgmask[lowest_point-crop_height:lowest_point, :]
                if(crop_fgmask.size>0):
                    if( (lowest_point-bottom)< crop_height and (right-left) > foot_width ):
                        # assume runner is coming from the right side (ccw running, filming from inside the track)
                        data.append([
                            left,
                            right,
                            abs(last_left-right),
                            last_left,
                            last_right,
                            bottom,
                            last_bottom,
                            frame_num,
                            last_frame_num
                        ])
                        last_left, last_right, last_bottom, last_frame_num = (left, right, bottom, frame_num)
                        cv2.imwrite("../media/"+video_nbmr+"/foot_placement/"+str(frame_num)+".png", fgmask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    cv2.imshow('cropped', crop_fgmask) # show mask video

        if not second_pass:
            cv2.imshow('mask', fgmask) # show mask video

        #cv2.imshow('original', frame) # show original video

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(data)>0 :
        data = np.array(data)
        print(data)
        jump = data[ np.argmax(data[:,2]) ] #find max delta between x values at bottom of slice
        # the jump/step starts at the last seen bottom left point (the front of the shoe coming from right)
        # the jump/step ends at the back of the foot which is the bottom right point
        print("jump occurred between pixels {} and  {} with a distance of approx {} pixels".format(
            jump[3], jump[1], jump[2]
        ))
        print('x,y start: {},{} - x,y end: {},{}'.format(
            jump[3],jump[6],jump[1],jump[5]
        ))
        jump_speed = 8.8/((jump[7]-jump[8])/fps) #distance / (frames in between jumps / fps)
        print('speed during jump (hardcoding 8.8 ft jump): {} ft/s'.format(
            jump_speed
        ))
        frame_path = "../media/"+video_nbmr+"/stats/default.jpg"
        cv2.imwrite(frame_path, stats_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        stats_frame = cv2.circle(stats_frame, (jump[1],jump[5]), radius=5, color=(50, 168, 82), thickness=-1)
        stats_frame = cv2.circle(stats_frame, (jump[3],jump[6]), radius=5, color=(50, 60, 168), thickness=-1)
        
        # font 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        # org 
        org = (50, 50) 
        # fontScale 
        fontScale = 1
        # Blue color in BGR 
        color = (255, 0, 0) 
        # Line thickness of 2 px 
        thickness = 2
        
        frame_path = "../media/"+video_nbmr+"/stats/stats.jpg"
        stats_frame = cv2.putText(
            stats_frame,
            'Jump speed: {} ft/second'.format(int(jump_speed)),
            org, font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA
        ) 
        cv2.imwrite(frame_path, stats_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        df = pd.DataFrame({
            'x_left': data[:, 0],
            'y_right': data[:, 1],
            'distance(last_toe_to_current_heel)' : data[:, 2],
            'last_x_left' : data[:, 3],
            'last_x_right' : data[:, 4],
            'y_bottom' : data[:, 5],
            'last_y_bottom' : data[:, 6],
            'frame' : data[:, 7],
            'last_frame' : data[:, 8],
        })
        frame_path = "../media/"+video_nbmr+"/stats/stats.csv"
        df.to_csv(frame_path, index = False)


    return lowest_point, first_frame, last_frame


# We can first identify people in the frame, then apply background segmentation within those bounds to avoid noise
# https://medium.com/@luanaebio/detecting-people-with-yolo-and-opencv-5c1f9bc6a810
# channels = 1 in yolov3.cfg for grayscale. If not using grayscale, change this back to 3
# read pre-trained model and config file
if detect_person:
    net = cv2.dnn.readNet('./models/yolov2-tiny.weights', './models/yolov2-tiny.cfg')

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
    confidence_threshold = 0.1
    person_class_id = 0
    width = image.shape[1]
    height = image.shape[0]
    max_box = None
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

                if max_box is None or box[2] * box[3] > max_box[2] * max_box[3]:
                    max_box = box

    return max_box

render_assets()
#lowest_point, first_frame, last_frame = run() #first pass
#_, _, _ = run(lowest_point, first_frame, last_frame, True)