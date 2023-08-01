import cv2
import numpy as np
import glob
import argparse
import os 

parser = argparse.ArgumentParser(description='PyTorch + mavsdk -- zero shot detection, tracking, and drone control')


parser.add_argument('--saved_images_dir', default=False, help='The path to save all semgentation/tracking frames')
parser.add_argument('--desired_height', default=240, type=int, help='desired_height resulution')
parser.add_argument('--desired_width', default=320, type=int, help='desired_width resulution')


args = parser.parse_args()

frameSize = (args.desired_width, args.desired_height)


for directory in ['SAM-result', 'Stream_segmenting', 'DINO-CLIP-result', 'Tracker-result', 'Stream_tracking', 'Detection']:
    directory_full_path = os.path.join(args.saved_images_dir,directory)
    out = cv2.VideoWriter(os.path.join(directory_full_path,'{}_video.avi'.format(directory)),cv2.VideoWriter_fourcc(*'DIVX'), 10, frameSize)
    fileidx = 0
    while 1:#filename in glob.glob('D:/images/*.jpg'):
        imgpath = os.path.join(directory_full_path,"{}.jpg".format(fileidx))
        if not os.path.exists(imgpath):
            break
        img = cv2.imread(imgpath)
        fileidx +=1
        out.write(img)

    out.release()
    print("done {}".format(directory))