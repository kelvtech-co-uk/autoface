import numpy as np
import cv2 as cv
import requests
# import argparse
import datetime
from yunet import YuNet
from sface import SFace

# parser = argparse.ArgumentParser()
# parser.add_argument("--input", help='still or video')
# args = parser.parse_args()

def visualize(target_image, faces1, query_image, faces2, matches, scores, target_size=[512, 512]): # target_size: (h, w)
    out1 = target_image.copy()
    out2 = query_image.copy()
    matched_box_color = (0, 255, 0)    # BGR
    mismatched_box_color = (0, 0, 255) # BGR

    # Resize to 256x256 with the same aspect ratio
    padded_out1 = np.zeros((target_size[0], target_size[1], 3)).astype(np.uint8)
    h1, w1, _ = out1.shape
    ratio1 = min(target_size[0] / out1.shape[0], target_size[1] / out1.shape[1])
    new_h1 = int(h1 * ratio1)
    new_w1 = int(w1 * ratio1)
    resized_out1 = cv.resize(out1, (new_w1, new_h1), interpolation=cv.INTER_LINEAR).astype(np.float32)
    top = max(0, target_size[0] - new_h1) // 2
    bottom = top + new_h1
    left = max(0, target_size[1] - new_w1) // 2
    right = left + new_w1
    padded_out1[top : bottom, left : right] = resized_out1

    # Draw bbox
    bbox1 = faces1[0][:4] * ratio1
    x, y, w, h = bbox1.astype(np.int32)
    cv.rectangle(padded_out1, (x + left, y + top), (x + left + w, y + top + h), matched_box_color, 2)

    # Resize to 256x256 with the same aspect ratio
    padded_out2 = np.zeros((target_size[0], target_size[1], 3)).astype(np.uint8)
    h2, w2, _ = out2.shape
    ratio2 = min(target_size[0] / out2.shape[0], target_size[1] / out2.shape[1])
    new_h2 = int(h2 * ratio2)
    new_w2 = int(w2 * ratio2)
    resized_out2 = cv.resize(out2, (new_w2, new_h2), interpolation=cv.INTER_LINEAR).astype(np.float32)
    top = max(0, target_size[0] - new_h2) // 2
    bottom = top + new_h2
    left = max(0, target_size[1] - new_w2) // 2
    right = left + new_w2
    padded_out2[top : bottom, left : right] = resized_out2

    # Draw bbox
    assert faces2.shape[0] == len(matches), "number of faces2 needs to match matches"
    assert len(matches) == len(scores), "number of matches needs to match number of scores"
    for index, match in enumerate(matches):
        bbox2 = faces2[index][:4] * ratio2
        x, y, w, h = bbox2.astype(np.int32)
        box_color = matched_box_color if match else mismatched_box_color
        cv.rectangle(padded_out2, (x + left, y + top), (x + left + w, y + top + h), box_color, 2)

        score = scores[index]
        text_color = matched_box_color if match else mismatched_box_color
        cv.putText(padded_out2, "{:.2f}".format(score), (x + left, y + top - 5), cv.FONT_HERSHEY_DUPLEX, 0.4, text_color)

    return np.concatenate([padded_out1, padded_out2], axis=1)

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
target_image = cv.imread("target.jpeg")

# Detect faces in target
detector.setInputSize([target_image.shape[1], target_image.shape[0]])
faces1 = detector.infer(target_image)
assert faces1.shape[0] > 0, 'Cannot find a face in {}'.format(target_image)

# snap_source = requests.get("http://192.168.1.99:1984/api/frame.jpeg?src=blackprostation", stream=True).raw
# query_image = np.asarray(bytearray(snap_source.read()), dtype="uint8")
# query_image = cv.imdecode(query_image, cv.IMREAD_COLOR)

# Load query video source
query_source_url = "rtsp://192.168.1.99:8554/blackprostation"
query_source = cv.VideoCapture(query_source_url)
# w = int(query_source.get(cv.CAP_PROP_FRAME_WIDTH))
# h = int(query_source.get(cv.CAP_PROP_FRAME_HEIGHT))
output = cv.VideoWriter('/mnt/cache/processing/result.mp4', cv.VideoWriter_fourcc(*'mp4v'), 25, (1024, 512), True)
try:
    while True:
        # read a frame from the query video source
        hasFrame, query_image = query_source.read()

        # Attempt to detect faces
        detector.setInputSize([query_image.shape[1], query_image.shape[0]])
        faces2 = detector.infer(query_image)
        
        # If no faces detected just move to the next frame and retry
        if faces2.shape[0] == 0:
            print('No faces deteceted in query source')
            continue
        
        # Match
        scores = []
        matches = []
        for face in faces2:
            result = recognizer.match(target_image, faces1[0][:-1], query_image, face[:-1])
            scores.append(result[0])
            matches.append(result[1])

        # Draw results
        image = visualize(target_image, faces1, query_image, faces2, matches, scores)
        
        output.write(image)
        
except KeyboardInterrupt:
    pass

print('\nVideo file written to disk.')





