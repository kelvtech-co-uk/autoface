
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
    
print('Running in', get_os())

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
              detection_time, 
              inference_time):
    matched_box_color = (0, 255, 0)    # BGR
    mismatched_box_color = (0, 0, 255) # BGR
        
    # Validate results
    assert query_faces.shape[0] == len(matches), "number of query_faces needs to match matches"
    assert len(matches) == len(scores), "number of matches needs to match number of scores"
    
    draw_text(query_image, text=f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}", pos=(0, 30))
    #cv.putText(query_image,f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}", (0, 30), cv.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

    if fps is not None:
        draw_text(query_image, text=f"FPS: {fps:.0f}", pos=(0, 45))
        #cv.putText(query_image, 'FPS: {:.2f}'.format(fps), (0, 45), cv.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

    # Only report detection time if there are faces found in the frame:
    if query_faces.shape[0] > 0:
        draw_text(query_image, text=f"Detection: {detection_time:.4f}ms", pos=(0, 60))
        #cv.putText(query_image,f"Detection: {detection_time:.4f}ms", (0, 60), cv.FONT_HERSHEY_SIMPLEX, 0.4, text_color)    

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

def resize_image(input_path, 
                 target_width):
    original_image = Image.open(input_path)

    # Check if the image has EXIF metadata
    if hasattr(original_image, '_getexif'):
        exif = original_image._getexif()
        if exif:
            for tag, label in ExifTags.TAGS.items():
                if label == 'Orientation':
                    orientation = tag
                    break
            if orientation in exif:
                if exif[orientation] == 3:
                    original_image = original_image.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    original_image = original_image.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    original_image = original_image.rotate(90, expand=True)
    
    aspect_ratio = original_image.width / original_image.height
    target_height = int(target_width / aspect_ratio)

    resized_image = original_image.resize((target_width, target_height), Image.LANCZOS)
    resized_image.save(str("_"+input_path))

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

collations = []
members = []
targets = []

# Map the directories located in /app/collations into a list called collations
for c in [c 
          for c in os.listdir('collations') 
          if os.path.isdir(os.path.join('collations', c))
          ]:
    collations.append('collations/' + c)

# For each mapped directory in the list collations, map the subdirectories into a list called members
for x in range(len(collations)):
    directory = collations[x]
    for m in [m 
              for m in os.listdir(directory) 
              if os.path.isdir(os.path.join(directory, m))
              ]:
        members.append(directory + "/" + m)

# For each mapped member directory in the list members, map the files into a list called targets
for x in range(len(members)):
    contents = members[x]
    if len([t 
            for t in os.listdir(contents) 
            if not t.startswith(".") and 
            os.path.isfile(os.path.join(contents, t))
            ]) > 1:
        raise ValueError("More than 1 unfiltered target file found in target directory:", contents,)
    for t in [t 
              for t in os.listdir(contents) 
              if not t.startswith(".") and 
              os.path.isfile(os.path.join(contents, t))
              ]:
        targets.append(contents + "/" + t)    

# Error if no target files are found
if len(targets) == 0:
    raise ValueError('No target files found, please ensure 1 unfiltered target file is placed into a collation/person folder')

# print('Collations:', collations)
# print('Members:', members)
print('Target files located:', targets)

# Load target image with basic pre-processing to enable detection
target_file = '/collations/middletons/kelvin/targetd.jpeg'
target_width = 500
target_image = cv.imread(target_file)
print('Pre (w & h):', target_file, target_image.shape[1], target_image.shape[0])
if target_image.shape[1] > 640:
    resize_image(target_file, target_width)
    target_image = cv.imread(str("_"+target_file))
    print('Post: (w & h):', target_image.shape[1], target_image.shape[0])
else:
    print('Original target file width within tolerance')

# Detect faces in target
detector.setInputSize([target_image.shape[1], target_image.shape[0]])
target_face = detector.infer(target_image)
if target_face.shape[0] == 0:
    sys.exit("No faces deteceted in any target files")

# Load query video source
query_source_url = "rtsp://192.168.1.99:8554/blackprostation"
#query_source_url = "rtsps://192.168.1.1:7441/EOEohGh0eoXIWf28"
#query_source_url = "event.mp4"

query_source = cv.VideoCapture(query_source_url)
query_source_width = int(query_source.get(cv.CAP_PROP_FRAME_WIDTH))
query_source_height = int(query_source.get(cv.CAP_PROP_FRAME_HEIGHT))

# ONLY FOR MY DEV ENV
if get_os() == "Linux":
    output = cv.VideoWriter('/mnt/cache/processing/result.mp4', cv.VideoWriter_fourcc(*'mp4v'), 25, (query_source_width, query_source_height), True)
elif get_os() == "Windows":
    output = cv.VideoWriter('result.mp4', cv.VideoWriter_fourcc(*'mp4v'), 25, (query_source_width, query_source_height), True)

fps = cv.TickMeter()
frame_skip = 2
try:
    while True:
        # read a frame from the query video source
        fps.start()
        hasFrame, query_image = query_source.read()
        if not hasFrame:
            sys.exit("Unable to grab a frame from query_source")

        # Attempt to detect faces
        det_start_time = time.time()
        detector.setInputSize([query_image.shape[1], query_image.shape[0]])
        query_faces = detector.infer(query_image)
        det_end_time = time.time()
        
        detection_time = det_end_time - det_start_time

        # Match faces to target
        scores = []
        matches = []
        infer_start_time = time.time()
        for face in query_faces:
            result = recognizer.match(target_image, target_face[0][:-1], query_image, face[:-1])
            scores.append(result[0])
            matches.append(result[1])
        
        infer_end_time = time.time()
            
        inference_time = infer_end_time - infer_start_time
        
        fps.stop()
        
        # Draw results
        image = visualize(query_image, query_faces, matches, scores, fps.getFPS(), detection_time, inference_time)
        
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





