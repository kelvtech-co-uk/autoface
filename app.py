
import numpy as np
import cv2 as cv
import sys
import platform
import datetime
import time
import os
from PIL import ExifTags

# Update the path variable to include addition module sub-directories
sys.path.append("yunet")
from yunet import YuNet
sys.path.append("sface")
from sface import SFace

# Ensure OpenCL is used
cv.ocl.setUseOpenCL(True)

# Instantiate YuNet & SFace
backendId = cv.dnn.DNN_BACKEND_OPENCV
targetId = cv.dnn.DNN_TARGET_CPU
detector = YuNet(modelPath='yunet/face_detection_yunet_2023mar.onnx',
                inputSize=[320, 320], 
                confThreshold=0.9,    #Usage: Set the minimum needed confidence for the model to identify a face. Smaller values may result in faster detection, but will limit accuracy.
                nmsThreshold=0.3,     #Usage: Suppress bounding boxes of iou >= nms_threshold.
                topK=5000,            #Usage: Keep top_k bounding boxes before NMS.
                backendId=backendId,
                targetId=targetId)
recognizer = SFace(modelPath='sface/face_recognition_sface_2021dec.onnx',
                disType=0,      #Usage: Distance type. \'0\': cosine, \'1\': norm_l1. Defaults to \'0\'
                backendId=backendId,
                targetId=targetId)

def get_os():
    system = platform.system()
    if system == "Windows":
        return "Windows"
    elif system == "Linux":
        return "Linux"
    else:
        return "Unknown"

def resize_image(target_image, target_width):

    # Check if the image has EXIF metadata
    if hasattr(target_image, '_getexif'):
        exif = target_image._getexif()
        if exif:
            for tag, label in ExifTags.TAGS.items():
                if label == 'Orientation':
                    orientation = tag
                    break
            if orientation in exif:
                if exif[orientation] == 3:
                    target_image = cv.rotate(target_image, cv.ROTATE_180)
                elif exif[orientation] == 6:
                    target_image = cv.rotate(target_image, cv.ROTATE_90_CLOCKWISE)
                elif exif[orientation] == 8:
                    target_image = cv.rotate(target_image, cv.ROTATE_90_COUNTERCLOCKWISE)
    
    # Resize the image keeping the aspect ratio
    original_height, original_width, _ = target_image.shape
    aspect_ratio = original_width / original_height
    target_height = int(target_width / aspect_ratio)
    
    return cv.resize(target_image, (target_width, target_height), interpolation=cv.INTER_LANCZOS4)

def draw_text(image, 
              text, 
              pos=(0, 0), 
              font=cv.FONT_HERSHEY_SIMPLEX, 
              font_scale=0.4, 
              text_color=(255, 255, 255), 
              font_thickness=1, 
              text_color_bg=(0, 0, 0), 
              border=1):
    x, y = pos
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv.rectangle(image, (x, y + border), (x + text_w, y - text_h - border), text_color_bg, -1)
    cv.putText(image, text, (x, y), font, font_scale, text_color, font_thickness)

    return

def decorate(image, queries, fps, inference_time):
    # queries: 0=face data, 1=match, 2=score, 3=target member
    matched_box_color = (0, 255, 0)    # BGR
    mismatched_box_color = (0, 0, 255) # BGR

    draw_text(query_image, text=f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}", pos=(0, 10))
    draw_text(image, text='{:.1}'.format(str(fps)) + ' fps', pos=(0, 22))
    draw_text(image, text='Inference: ' + '{:.2}'.format(inference_time) + 'ms', pos=(0, 34))

    for query in queries:
        detection_confidence = query[0][14]
        bbox = query[0][0:4]
        x, y, w, h = bbox.astype(np.int32)
        match = query[1]
        box_color = matched_box_color if match else mismatched_box_color
        
        cv.rectangle(image, (x, y), (x + w, y + h), box_color, 2)
        draw_text(image, text= '{:.2%}'.format(detection_confidence), pos=((x + 2), (y + 12)))
        if match == 1:
            draw_text(image, text='{:.2%}'.format(query[2]) + ' ' + query[3], pos=((x + 2), (y + 24)))

    return image

def build_targets():
    # Parse the collations_root folder and all its contents and resize as necessary passable target images
    collations_root = 'collations'
    # collations = []
    # members = []
    targets = []    # targets: 0=collation, 1=member, 2=file, 3=image data, 4=face data
    target_width = 500
    for collation in os.listdir(collations_root):   # Find the collation folders in the collation_root directory
        if os.path.isdir(os.path.join(collations_root, collation)):     # Ignore anything thats not a directory
            for member in os.listdir(os.path.join(collations_root, collation)):     # Find the member folders in each collation folder
                if os.path.isdir(os.path.join(collations_root, collation, member)):     # Ignore anything thats not a directory
                    for file in os.listdir(os.path.join(collations_root, collation, member)):   # Find the files in the member folders by...
                        if not file.startswith(".") and os.path.isfile(os.path.join(collations_root, collation, member, file)):     # ...ignoring anything prefixed with a period (.) or that isn't a file
                            targets.append([collation, member, file, None, None])   # Append the indivdual lists into the object 'target' creating a nested list object with placeholder positions
                            targets[-1][3] = cv.imread(os.path.join(collations_root, targets[-1][0], targets[-1][1], targets[-1][2]))   # Rebuild the file path from the list and read in the image
                    #
                    # Need some error handling logic for files cv2.imread fails to process such as targetc.jpeg 
                    #
                            if targets[-1][3].shape[1] > target_width:  # Check the width of the image is within tollerance
                                targets[-1][3] = resize_image(targets[-1][3], target_width)     # If not resize the image in-situ
                                print('Resized:', os.path.join(collations_root, targets[-1][0], targets[-1][1], targets[-1][2]))
                            detector.setInputSize([targets[-1][3].shape[1], targets[-1][3].shape[0]]) # Setup the face detector
                            face_test = detector.infer(targets[-1][3])  # Attempt to detect faces in the target image
                            if face_test.shape[0] == 1:     # If there is a face...
                                targets[-1][4] = face_test  # ...save it back into the nested list alongside the source data
                            elif face_test.shape[0] > 1:
                                raise Exception('More than 1 face detected in target file: ' + os.path.join(collations_root, targets[-1][0], targets[-1][1], targets[-1][2]))


    for index, target in enumerate(targets):
        print('index:', index, 'collation:', target[0], 'member:', target[1], 'file:', target[2], 'face:', target[4].shape)

    # Error if no target files are found
    if len(targets) == 0:
        raise Exception("No target files found. Please ensure one unfiltered target file is placed into a collation/person folder.")

    if targets[-1][4].shape[0] == 0:
        raise Exception("No faces deteceted in any target files")
    
    return targets

def frameTypeWebcam():
    query_source = cv.VideoCapture(0)
    # query_source_width = int(query_source.get(cv.CAP_PROP_FRAME_WIDTH))
    # query_source_height = int(query_source.get(cv.CAP_PROP_FRAME_HEIGHT))

    return query_source

def frameTypeRTSP():
    #query_source_url = "rtsp://192.168.1.99:8554/blackprostation"
    query_source_url = "rtsp://localhost:8554/boys_low"
    query_source = cv.VideoCapture(query_source_url)
    # query_source_width = int(query_source.get(cv.CAP_PROP_FRAME_WIDTH))
    # query_source_height = int(query_source.get(cv.CAP_PROP_FRAME_HEIGHT))

    return query_source

def frameTypeVideo():
    query_source_url = "event.mp4"
    query_source = cv.VideoCapture(query_source_url)

    return query_source

def frameTypeImageRead():
    query_source = 'family.jpg'
    query_image = cv.imread(query_source)
    query_image = resize_image(query_image, 1000)

    return query_image

def frameRead(query_source):
    hasFrame, query_image = query_source.read()
    if not hasFrame:
        failFrame += 1
        if failFrame == 30:
            raise Exception('Failed to grab a frame after {failFrame} consequtive tries')
    else:
        failFrame = 0 

    return query_image

print('Running in ', get_os())
targets = build_targets()  
query_source = frameTypeRTSP()

fps = cv.TickMeter()
frame_skip = 3
failFrame = 0

while True:
    queries = []

    fps.start()
    query_image = frameRead(query_source)
    cv.imshow('query_source', query_image)
    #query_image = frameTypeImageRead()
    
    # Attempt to detect faces
    infer_start_time = time.time()
    detector.setInputSize([query_image.shape[1], query_image.shape[0]]) 
    faces = detector.infer(query_image)

    for f_index, face in enumerate(faces):
        queries.insert(f_index, [face, None, None, None])
        for target in targets:
            # targets: 0=collation, 1=member, 2=file, 3=image data, 4=face data
            # queries: 0=face data, 1=match, 2=score, 3=target member
            # recognizer.match(target_img, target_face, query_img, query_face)
            result = recognizer.match(target[3], target[4], query_image, queries[f_index][0])
            score = result[0]
            match = result[1]
            if match == 1:
                if queries[f_index][2] is None or score > queries[f_index][2]:
                    queries.pop(f_index)
                    queries.insert(f_index, [face, result[1], score, target[1]])

    infer_end_time = time.time()
    inference_time = infer_end_time - infer_start_time
    fps.stop()
    
    # Draw results
    decorated_image = decorate(query_image, queries, fps.getFPS(), inference_time)

    cv.imshow('autoface', decorated_image)
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()
query_source.release()

print('\nProgram exited')





