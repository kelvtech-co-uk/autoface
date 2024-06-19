import numpy as np
import cv2 as cv
# import argparse
import sys
import platform
from yunet import YuNet
from sface import SFace

# parser = argparse.ArgumentParser()
# parser.add_argument("--input", help='still or video')
# args = parser.parse_args()

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

def visualize(query_image, query_faces, matches, scores): # target_size: (h, w)
    matched_box_color = (0, 255, 0)    # BGR
    mismatched_box_color = (0, 0, 255) # BGR

    # Validate results
    assert query_faces.shape[0] == len(matches), "number of query_faces needs to match matches"
    assert len(matches) == len(scores), "number of matches needs to match number of scores"
    
    # Draw bbox
    for index, match in enumerate(matches):
        bbox = query_faces[index][:4]
        x, y, w, h = bbox.astype(np.int32)
        box_color = matched_box_color if match else mismatched_box_color
        cv.rectangle(query_image, (x, y), (x + w, y + h), box_color, 2)

        score = scores[index]
        text_color = matched_box_color if match else mismatched_box_color
        cv.putText(query_image, "{:.2f}".format(score), (x, y - 5), cv.FONT_HERSHEY_DUPLEX, 0.4, text_color)

        return query_image

# Instantiate YuNet & SFace
detector = YuNet(modelPath='face_detection_yunet_2023mar.onnx',
                inputSize=[320, 320], 
                confThreshold=0.9,    #Usage: Set the minimum needed confidence for the model to identify a face. Smaller values may result in faster detection, but will limit accuracy.
                nmsThreshold=0.3,     #Usage: Suppress bounding boxes of iou >= nms_threshold.
                topK=5000,            #Usage: Keep top_k bounding boxes before NMS.
                backendId=cv.dnn.DNN_BACKEND_OPENCV,
                targetId=cv.dnn.DNN_TARGET_CPU)

recognizer = SFace(modelPath='face_recognition_sface_2021dec.onnx',
                    disType=0,      #Usage: Distance type. \'0\': cosine, \'1\': norm_l1. Defaults to \'0\'
                    backendId=cv.dnn.DNN_BACKEND_OPENCV,
                    targetId=cv.dnn.DNN_TARGET_CPU)

# Load target image
target_image = cv.imread("target2.jpeg")
target_image = cv.resize(target_image, (180,320), interpolation=cv.INTER_LINEAR)

# Detect faces in target
detector.setInputSize([target_image.shape[1], target_image.shape[0]])
target_face = detector.infer(target_image)
if target_face.shape[0] == 0:
    sys.exit("No faces deteceted in query source")

# Load query video source
query_source_url = "rtsp://192.168.1.99:8554/blackprostation"
query_source = cv.VideoCapture(query_source_url)
query_source_width = int(query_source.get(cv.CAP_PROP_FRAME_WIDTH))
query_source_hight = int(query_source.get(cv.CAP_PROP_FRAME_HEIGHT))

if get_os() == "Linux":
    output = cv.VideoWriter('/mnt/cache/processing/result.mp4', cv.VideoWriter_fourcc(*'mp4v'), 25, (query_source_width, query_source_hight), True)

try:
    while True:
        # read a frame from the query video source
        hasFrame, query_image = query_source.read()

        # Attempt to detect faces
        detector.setInputSize([query_image.shape[1], query_image.shape[0]])
        query_faces = detector.infer(query_image)
        
        # If no faces detected just move to the next frame and retry
        if query_faces.shape[0] == 0:
            #print('No faces deteceted in query source')
            continue
        
        # Match
        scores = []
        matches = []
        for face in query_faces:
            result = recognizer.match(target_image, target_face[0][:-1], query_image, face[:-1])
            scores.append(result[0])
            matches.append(result[1])

        # Draw results
        image = visualize(query_image, query_faces, matches, scores)
        
        if get_os() == "Linux":
            output.write(image)
        elif get_os() == "Windows":
            cv.imshow('autoface', image)
            if cv.waitKey(1) == ord('q'):
                break
        
except KeyboardInterrupt:
    pass

if get_os() == "Linux":
    output.release()
elif get_os() == "Windows":
    cv.destroyAllWindows()

query_source.release()
print('\nProgram exited')





