
import cv2 as cv
import threading

class Camera(object):
    def __init__(self, source):
        self._source = source
        self.query_image_lock = threading.Lock()
        self.query_image = None
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

        self.query_source = cv.VideoCapture(self._source_url, self._apiPref)
        self.query_source.set(cv.CAP_PROP_BUFFERSIZE, 1)
        self.query_source.set(cv.CAP_PROP_FPS, 20)

        if self.query_source.isOpened():
            print('Query source successfully opened.')
            return
        else:
            raise Exception('Unable to open query source')
    
    def read_frames(self):
        _failFrame = 0
        while self.query_source.isOpened():
            with self.query_image_lock:
                hasFrame, self.query_image = self.query_source.read()
            if not hasFrame:
                _failFrame += 1
                if _failFrame == 15:
                    raise Exception(f"Failed to grab a frame after {_failFrame} consequtive tries")
                continue
            else:
                _failFrame = 0
                # cv.imshow('autoface', query_image)
                # if cv.waitKey(1) == ord('q'):
                #     self.close()
    
    def show_frame(self):
        while True:
            with self.query_image_lock:
                cv.imshow('autoface', self.query_image)
            if cv.waitKey(1) == ord('q'):
                self.query_source.release()
                break

    def close(self):
        if self.query_source.isOpened():
            cv.destroyAllWindows()
            self.query_source.release()
            print('\nQuery source released')

if __name__ == '__main__':
    capture = Camera('cam')

    capture_read_frame_thread = threading.Thread(target=capture.read_frames, args=(), daemon=False)
    capture_read_frame_thread.start()
    
    print('starting display')
    capture_show_frame_thread = threading.Thread(target=capture.show_frame, args=(), daemon=False)
    capture_show_frame_thread.start()
    
    print('joining ...')
    capture_show_frame_thread.join()
    capture_read_frame_thread.join()
    
    print('closing down')
    capture.close()

    print('\nProgram exited')
