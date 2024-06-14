import numpy as np
import cv2 as cv
import requests
from yunet import YuNet

def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    output = image.copy()
    landmark_color = [
        (255,   0,   0), # right eye
        (  0,   0, 255), # left eye
        (  0, 255,   0), # nose tip
        (255,   0, 255), # right mouth corner
        (  0, 255, 255)  # left mouth corner
    ]

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    for det in results:
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

        conf = det[-1]
        cv.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(output, landmark, 2, landmark_color[idx], 2)
    
    return output

# Instantiate YuNet
model = YuNet(modelPath='face_detection_yunet_2023mar.onnx',
                inputSize=[320, 320], 
                confThreshold=0.9,    #Usage: Set the minimum needed confidence for the model to identify a face. Smaller values may result in faster detection, but will limit accuracy.
                nmsThreshold=0.3,     #Usage: Suppress bounding boxes of iou >= nms_threshold.
                topK=5000,            #Usage: Keep top_k bounding boxes before NMS.
                backendId=cv.dnn.DNN_BACKEND_OPENCV,
                targetId=cv.dnn.DNN_TARGET_CPU)

# If input is an image
# if args.input is not None:
snap_source = requests.get("http://192.168.1.99:1984/api/frame.jpeg?src=blackprostation", stream=True).raw
image = np.asarray(bytearray(snap_source.read()), dtype="uint8")
image = cv.imdecode(image, cv.IMREAD_COLOR)
h, w, _ = image.shape

# Inference
model.setInputSize([w, h])
results = model.infer(image)

# Print results
print('{} faces detected.'.format(results.shape[0]))
for idx, det in enumerate(results):
    print('{}: {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}'.format(
        idx, *det[:-1])
    )

# Draw results on the input image
image = visualize(image, results)

# Save results if save is true
cv.imwrite('result.jpg', image)

    # else: # Omit input to call default camera
    #     deviceId = 0
    #     cap = cv.VideoCapture(deviceId)
    #     w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    #     h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    #     model.setInputSize([w, h])

    #     tm = cv.TickMeter()
    #     while cv.waitKey(1) < 0:
    #         hasFrame, frame = cap.read()
    #         if not hasFrame:
    #             print('No frames grabbed!')
    #             break

    #         # Inference
    #         tm.start()
    #         results = model.infer(frame) # results is a tuple
    #         tm.stop()

    #         # Draw results on the input image
    #         frame = visualize(frame, results, fps=tm.getFPS())

    #         # Visualize results in a new Window
    #         cv.imshow('YuNet Demo', frame)

    #         tm.reset()