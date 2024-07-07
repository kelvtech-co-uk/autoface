import cv2 as cv
import multiprocessing

class Camera(object):
    def __init__(self, source):
        self._source = source
        self._capture_queue = multiprocessing.Queue()

        match self._source:
            case 'cam':
                self._source_url = 0
                self._apiPref = cv.CAP_DSHOW
            case 'boysr':
                self._source_url = 'rtsps://192.168.1.1:7441/xhw7D6R7BR8NP5vg'
                self._apiPref = cv.CAP_FFMPEG
            case 'frontd':   
                self._source_url = 'rtsps://192.168.1.1:7441/EOEohGh0eoXIWf28'
                self._apiPref = cv.CAP_FFMPEG
            case 'video':
                self._source_url = 'event.mp4'
                self._apiPref = cv.CAP_FFMPEG
    
    def read_frames(self, _capture_queue, quit_read_frame):
        self._capture_source = cv.VideoCapture(self._source_url, self._apiPref)
        self._capture_source.set(cv.CAP_PROP_BUFFERSIZE, 1)
        self._capture_source.set(cv.CAP_PROP_FPS, 20)

        if self._capture_source.isOpened():
            print('Capture object successfully started')
        else:
            raise Exception('Unable to start capture object')

        _failFrame = 0
        while not quit_read_frame.is_set():
            _hasFrame, _capture_image = self._capture_source.read()
            _capture_queue.put(_capture_image)
            if not _hasFrame:
                _failFrame += 1
                if _failFrame == 15:
                    raise Exception(f"Failed to grab a frame after {_failFrame} consequtive tries")
                continue
            else:
                _failFrame = 0

        self._capture_source.release()
        self._capture_queue.cancel_join_thread()

    def show_frame(self, _capture_queue):
        while True:
            cv.imshow('autoface', _capture_queue.get())
            if cv.waitKey(1) == ord('q'):
                cv.destroyAllWindows()
                break

if __name__ == '__main__':
    print('Starting capture object')
    capture = Camera('cam')

    # Create the event to use to stop the program
    quit_read_frame = multiprocessing.Event()

    print('Starting read_frames process')
    capture_read_frame_process = multiprocessing.Process(target=capture.read_frames, args=(capture._capture_queue, quit_read_frame,))
    capture_read_frame_process.start()
    
    print('Starting show_frames process')
    capture_show_frame_process = multiprocessing.Process(target=capture.show_frame, args=(capture._capture_queue,))
    capture_show_frame_process.start()

    
    capture_show_frame_process.join()   # Should complete when 'q' is typed when the cv.imshow window is open and active.
    
    print('Program exiting')
    quit_read_frame.set()     # This should then break the loop in the read_frames function of the Camera object
    capture_read_frame_process.join()   # This should then exit as the queue.cancel_join_thread() method has been called breaking the queue/process dependency for flushing
    
    print('\nProgram exited')
