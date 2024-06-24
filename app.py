
import numpy as np
import cv2 as cv
import sys
import platform
import datetime
import time
import os
from PIL import Image, ExifTags

# Update the path variable to include addition module sub-directories
sys.path.append("yunet")
from yunet import YuNet
sys.path.append("sface")
from sface import SFace

# Ensure OpenCL is used
cv.ocl.setUseOpenCL(True)

def get_os():
    system = platform.system()
    if system == "Windows":
        return "Windows"
    elif system == "Linux":
        return "Linux"
    else:
        return "Unknown"
    
print('Running in', get_os(),'\n')

def draw_text(query_image, 
              text, 
              pos=(0, 0), 
              font=cv.FONT_HERSHEY_SIMPLEX, 
              font_scale=0.5, 
              text_color=(255, 255, 255), 
              font_thickness=1, 
              text_color_bg=(0, 0, 0), 
              border=1):
    x, y = pos
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv.rectangle(query_image, (x, y + border), (x + text_w, y - text_h - border), text_color_bg, -1)
    cv.putText(query_image, text, (x, y), font, font_scale, text_color, font_thickness)

    return #text_size

def visualize(query_image, 
              query_faces, 
              matches, 
              scores, 
              fps,
              inference_time):
    matched_box_color = (0, 255, 0)    # BGR
    mismatched_box_color = (0, 0, 255) # BGR
        
    # Validate results
    #assert query_faces.shape[0] == len(matches), "number of query_faces needs to match matches"
    #assert len(matches) == len(scores), "number of matches needs to match number of scores"
    
    draw_text(query_image, text=f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}", pos=(0, 30))
    #cv.putText(query_image,f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}", (0, 30), cv.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

    if fps is not None:
        draw_text(query_image, text=f"FPS: {fps:.0f}", pos=(0, 45))
        #cv.putText(query_image, 'FPS: {:.2f}'.format(fps), (0, 45), cv.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

    # Draw bbox
    for index, match in enumerate(matches):
        bbox = query_faces[index][:4]
        x, y, w, h = bbox.astype(np.int32)
        box_color = matched_box_color if match else mismatched_box_color
        cv.rectangle(query_image, (x, y), (x + w, y + h), box_color, 2)

        score = scores[index]
        text_color = matched_box_color if match else mismatched_box_color
        draw_text(query_image, text=f"Inference: {inference_time:.4f}ms", pos=(0, 75))
        cv.putText(query_image, "{:.2f}".format(score), (x, y - 5), cv.FONT_HERSHEY_DUPLEX, 0.4, text_color)

    return query_image

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

# Instantiate YuNet & SFace
backendId = cv.dnn.DNN_BACKEND_OPENCV
#backendId = cv.dnn.DNN_BACKEND_INFERENCE_ENGINE
targetId = cv.dnn.DNN_TARGET_CPU
#targetId = cv.dnn.DNN_TARGET_OPENCL
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

collations_root = 'collations'
collations = []
members = []
targets = []    # Positions: 0=collation, 1=member, 2=file, 3=image data, 4=face data
target_width = 500
for collation in os.listdir(collations_root):   # Find the collation folders in the collation_root directory
    if os.path.isdir(os.path.join(collations_root, collation)):     # Ignore anything thats not a directory
        for member in os.listdir(os.path.join(collations_root, collation)):     # Find the member folders in each collation folder
            if os.path.isdir(os.path.join(collations_root, collation, member)):     # Ignore anything thats not a directory
                for file in os.listdir(os.path.join(collations_root, collation, member)):   # Find the files in the member folders by...
                    if not file.startswith(".") and os.path.isfile(os.path.join(collations_root, collation, member, file)):     # ...ignoring anything prefixed with a "." or isn't a file
                        targets.append([collation, member, file, None, None])   # Append the indivdual lists into the object target creating a nested list object with placeholder positions
                        targets[-1][3] = cv.imread(os.path.join(collations_root, targets[-1][0], targets[-1][1], targets[-1][2]))   # Rebuild the file path from the list and read in the image
                        if targets[-1][3].shape[1] > target_width:  # Check the width of the image is within tollerance
                            targets[-1][3] = resize_image(targets[-1][3], target_width)     # If not resize the image in-situ
                            print('Resized:', os.path.join(collations_root, targets[-1][0], targets[-1][1], targets[-1][2]))
                        detector.setInputSize([targets[-1][3].shape[1], targets[-1][3].shape[0]]) # Setup the face detector
                        face_test = detector.infer(targets[-1][3])  # Attempt to detect faces in the target image
                        if face_test.shape[0] != 0:     # If there is a face...
                            targets[-1][4] = face_test  # ...save it back into the nest list alongside the source data

# Error if no target files are found
if len(targets) == 0:
    raise Exception("No target files found. Please ensure one unfiltered target file is placed into a collation/person folder.")

if targets[-1][4].shape[0] == 0:
    raise Exception("No faces deteceted in any target files")

# Load query video source
query_source_url = "rtsp://192.168.1.99:8554/blackprostation"
#query_source_url = "rtsps://192.168.1.1:7441/EOEohGh0eoXIWf28"
#query_source_url = "event.mp4"
sys.exit()
query_source = cv.VideoCapture(query_source_url)
query_source_width = int(query_source.get(cv.CAP_PROP_FRAME_WIDTH))
query_source_height = int(query_source.get(cv.CAP_PROP_FRAME_HEIGHT))

# ONLY FOR MY DEV ENV
if get_os() == "Linux":
    output = cv.VideoWriter('/mnt/cache/processing/result.mp4', cv.VideoWriter_fourcc(*'mp4v'), 25, (query_source_width, query_source_height), True)
elif get_os() == "Windows":
    output = cv.VideoWriter('result.mp4', cv.VideoWriter_fourcc(*'mp4v'), 25, (query_source_width, query_source_height), True)

fps = cv.TickMeter()
frame_skip = 10
query_faces = []
try:
    while True:
        # read a frame from the query video source
        fps.start()
        hasFrame, query_image = query_source.read()
        if not hasFrame:
            sys.exit("Unable to grab a frame from query_source")

        # Attempt to detect faces
        infer_start_time = time.time()
        detector.setInputSize([query_image.shape[1], query_image.shape[0]])
        if detector.infer(query_image).shape[0] != 0:
            query_faces.append(detector.infer(query_image))
            #print(query_faces[-1])
        else:
            #print('No query faces detected in frame')
            continue

        # Match faces to target
        scores = []
        matches = []
        index = 0
        for query_face in query_faces:
            result = recognizer.match(target_images[0], target_faces[0][:-1], query_image, query_face)
            scores.append(result[0])
            matches.append(result[1])
        
            # for target_face in target_faces:
            #     result = recognizer.match(target_images[0], target_faces[0][:-1], query_image, query_face[-1])
            #     index += 1
            #     scores.append(result[0])
            #     matches.append(result[1])
            #     #print('Score:', scores[-1], 'Match:', matches[-1])
                
            infer_end_time = time.time()
            inference_time = infer_end_time - infer_start_time
            fps.stop()

                # Draw results
            image = visualize(query_image, query_faces, matches, scores, fps.getFPS(), inference_time)

        if get_os() == "Linux":
            output.write(image)
        elif get_os() == "Windows":
            cv.imshow('autoface', image)
            if cv.waitKey(1) == ord('q'):
                break
        
        for _ in range(frame_skip - 1):
            query_source.read()

except KeyboardInterrupt:
    pass

if get_os() == "Linux":
    output.release()
elif get_os() == "Windows":
    cv.destroyAllWindows()

query_source.release()
print('\nProgram exited')





