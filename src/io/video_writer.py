import cv2
import threading
from queue import Queue

class ThreadedVideoWriter:

    def __init__(self, dst, width, height, fps=30, queue_size=256):
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.writer = cv2.VideoWriter(dst, fourcc, fps, (width, height))
        self.Q = Queue(maxsize=queue_size)
        self.stopped = False

    def start(self):
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()
        return self

    def update(self):
        while True:
            if self.stopped and self.Q.empty():
                self.writer.release()
                return
            if not self.Q.empty():
                frame = self.Q.get()
                self.writer.write(frame)
            else:
                import time
                time.sleep(0.01)

    def write(self, frame):
        self.Q.put(frame)

    def stop(self):
        self.stopped = True
        self.t.join()