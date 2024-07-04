import cv2 as cv
import multiprocessing
import time

class Camera(object):
    def __init__(self, source):
        self._source = source
        self.image_queue = multiprocessing.Queue()

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
    
    def read_frames(self, queue):
        self.query_source = cv.VideoCapture(self._source_url, self._apiPref)
        self.query_source.set(cv.CAP_PROP_BUFFERSIZE, 1)
        self.query_source.set(cv.CAP_PROP_FPS, 20)

        if self.query_source.isOpened():
            print('Query source successfully opened.')
        else:
            raise Exception('Unable to open query source')
    
        _failFrame = 0
        while self.query_source.isOpened():
            hasFrame, image = self.query_source.read()
            if not hasFrame:
                _failFrame += 1
                if _failFrame == 15:
                    raise Exception(f"Failed to grab a frame after {_failFrame} consequtive tries")
                continue
            else:
                _failFrame = 0
                queue.put(image)

    def show_frame(self, queue):
        while True:
            image = queue.get()
            cv.imshow('autoface', image)
            if cv.waitKey(1) == ord('q'):
                cv.destroyAllWindows()
                
                
                break           

if __name__ == '__main__':
    capture = Camera('cam')

    capture_read_frame_process = multiprocessing.Process(target=capture.read_frames, args=(capture.image_queue,))
    capture_read_frame_process.start()

    print('starting display')
    capture_show_frame_process = multiprocessing.Process(target=capture.show_frame, args=(capture.image_queue,))
    capture_show_frame_process.start()

    capture_read_frame_process.join()
    capture_show_frame_process.join()
    capture.query_source.release()
    print('\nQuery source released')
    print('\nProgram exited')
