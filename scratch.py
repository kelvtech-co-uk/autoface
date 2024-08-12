import cv2 as cv
import numpy as np
import multiprocessing
import ctypes

# cap = cv.VideoCapture(0, cv.CAP_DSHOW)
# print('width:', cap.get(cv.CAP_PROP_FRAME_WIDTH))
# print('height:', cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# print('channels:', cap.get(cv.CAP_PROP_CHANNEL))
# print('format:', cap.get(cv.CAP_PROP_FORMAT))
# _, image = cap.read()
# print(image.shape)
# cap.release()

def process_frame(frame, shared_array):
    # Process the frame (e.g., apply some filter)
    processed_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Copy the processed frame data to the shared array
    shared_array[:] = processed_frame.flatten()

def main():
    # Initialize video capture
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv.CAP_PROP_FPS, 30)

    # Create a shared array to store processed frame data
    shape = (480, 640, 3)  # Assuming frame size
    shared_array = multiprocessing.Array(ctypes.c_uint16, shape[0] * shape[1] * shape[2], lock=False)

    # Create a pool of worker processes
    num_processes = 8
    with multiprocessing.Pool(processes=num_processes) as pool:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame in parallel
            pool.apply_async(process_frame, args=(frame, shared_array))

            # Display the original frame
            cv.imshow("Original Frame", frame)

            # Display the processed frame (from shared array)
            processed_frame = np.array(shared_array).reshape(frame.shape).astype(np.uint8)
            cv.imshow("Processed Frame", processed_frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
