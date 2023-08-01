# --------------------------------------------------------
# Camera sample code for Tegra X2/X1
#
# This program could capture and display video from
# IP CAM, USB webcam, or the Tegra onboard camera.
# Refer to the following blog post for how to set up
# and run the code:
#   https://jkjung-avt.github.io/tx2-camera-with-python/
#
# Written by JK Jung <jkjung13@gmail.com>
# --------------------------------------------------------


import sys
import argparse
import subprocess

import cv2
import time
from threading import Thread




class ThreadedCamera(object):
    def __init__(self, src=0, fps=0):
       
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
       
        # FPS = 1/X
        # X = desired FPS
        self.FPS = False
        self.FPS_MS = False
        if fps > 0:
            self.FPS = 1/fps
            self.FPS_MS = int(self.FPS * 1000)
            
        # Start frame retrieval thread
        self.read_once = False
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        print("done setting video")
        
    def update(self):
    
        while True:
            if self.capture.isOpened():
                (self.status, self.tmp_frame) = self.capture.read()
                if self.status:
                    self.read_once = True
                    self.frame = self.tmp_frame
                else:
                    print("Camera report: No new frames, read status: ", self.status)
            
            if self.FPS: 
                time.sleep(self.FPS)
    def read(self):
        if self.read_once: 
            return True, self.frame
        else:
            return False,None

#if __name__ == '__main__':
   
