import cv2
import threading
from queue import Queue

class ThreadedVideoReader:

    def __init__(self, src, queue_size=256):
        self.stream = cv2.VideoCapture(src)
        self.stopped = False
        self.Q = Queue(maxsize=queue_size)
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            if not self.Q.full():
                grabbed, frame = self.stream.read()
                if not grabbed:
                    self.stop()
                    return
                self.Q.put(frame)
            else:
                import time
                time.sleep(0.01)

    def read(self):
        return self.Q.get()

    def more(self):
        return self.Q.qsize() > 0 or not self.stopped

    def stop(self):
        self.stopped = True
        self.stream.release()