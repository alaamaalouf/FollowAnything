
import os

import cv2

from SegTracker import SegTracker

from model_args import aot_args,sam_args,segtracker_args

from PIL import Image

from aot_tracker import _palette

import numpy as np

import torch

import imageio

import matplotlib.pyplot as plt

from scipy.ndimage import binary_dilation

import gc

def save_prediction(pred_mask,output_dir,file_name):

    save_mask = Image.fromarray(pred_mask.astype(np.uint8))

    save_mask = save_mask.convert(mode='P')

    save_mask.putpalette(_palette)

    save_mask.save(os.path.join(output_dir,file_name))

def colorize_mask(pred_mask):

    save_mask = Image.fromarray(pred_mask.astype(np.uint8))

    save_mask = save_mask.convert(mode='P')

    save_mask.putpalette(_palette)

    save_mask = save_mask.convert(mode='RGB')

    return np.array(save_mask)

def draw_mask(img, mask, alpha=0.5, id_countour=False):

    img_mask = np.zeros_like(img)

    img_mask = img

    if id_countour:

        # very slow ~ 1s per image

        obj_ids = np.unique(mask)

        obj_ids = obj_ids[obj_ids!=0]

        for id in obj_ids:

            # Overlay color on  binary mask

            if id <= 255:

                color = _palette[id*3:id*3+3]

            else:

                color = [0,0,0]

            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)

            binary_mask = (mask == id)

            # Compose image

            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask

            img_mask[countours, :] = 0

    else:

        binary_mask = (mask!=0)

        countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask

        foreground = img*(1-alpha)+colorize_mask(mask)*alpha

        img_mask[binary_mask] = foreground[binary_mask]

        img_mask[countours,:] = 0

        

    return img_mask.astype(img.dtype)

video_name = 'cell'

io_args = {

    'input_video': f'./assets/{video_name}.mp4',

    'output_mask_dir': f'./assets/{video_name}_masks', # save pred masks

    'output_video': f'./assets/{video_name}_seg.mp4', # mask+frame vizualization, mp4 or avi, else the same as input video

    'output_gif': f'./assets/{video_name}_seg.gif', # mask visualization

}

sam_args['generator_args'] = {

        'points_per_side': 30,

        'pred_iou_thresh': 0.8,

        'stability_score_thresh': 0.9,

        'crop_n_layers': 1,

        'crop_n_points_downscale_factor': 2,

        'min_mask_region_area': 200,

    }

cap = cv2.VideoCapture(io_args['input_video'])

frame_idx = 0

segtracker = SegTracker(segtracker_args,sam_args,aot_args)

segtracker.restart_tracker()

with torch.cuda.amp.autocast():

    while cap.isOpened():

        ret, frame = cap.read()

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        pred_mask = segtracker.seg(frame)

        torch.cuda.empty_cache()

        obj_ids = np.unique(pred_mask)

        obj_ids = obj_ids[obj_ids!=0]

        print("processed frame {}, obj_num {}".format(frame_idx,len(obj_ids)),end='\n')

        break

    cap.release()

    init_res = draw_mask(frame,pred_mask,id_countour=False)

    plt.figure(figsize=(10,10))

    plt.axis('off')

    plt.imshow(init_res)

    plt.show()

    plt.figure(figsize=(10,10))

    plt.axis('off')

    plt.imshow(colorize_mask(pred_mask))

    plt.show()

    del segtracker

    torch.cuda.empty_cache()

    gc.collect()

# For every sam_gap frames, we use SAM to find new objects and add them for tracking

# larger sam_gap is faster but may not spot new objects in time

segtracker_args = {

    'sam_gap': 5, # the interval to run sam to segment new objects

    'min_area': 200, # minimal mask area to add a new mask as a new object

    'max_obj_num': 255, # maximal object number to track in a video

    'min_new_obj_iou': 0.8, # the area of a new object in the background should > 80% 

}

# source video to segment

cap = cv2.VideoCapture(io_args['input_video'])

fps = cap.get(cv2.CAP_PROP_FPS)

# output masks

output_dir = io_args['output_mask_dir']

if not os.path.exists(output_dir):

    os.makedirs(output_dir)

pred_list = []

masked_pred_list = []

torch.cuda.empty_cache()

gc.collect()

sam_gap = segtracker_args['sam_gap']

frame_idx = 0

segtracker = SegTracker(segtracker_args,sam_args,aot_args)

segtracker.restart_tracker()

with torch.cuda.amp.autocast():

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:

            break

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        if frame_idx == 0:

            pred_mask = segtracker.seg(frame)

            torch.cuda.empty_cache()

            gc.collect()

            segtracker.add_reference(frame, pred_mask)

        elif (frame_idx % sam_gap) == 0:

            seg_mask = segtracker.seg(frame)

            torch.cuda.empty_cache()

            gc.collect()

            track_mask = segtracker.track(frame)

            # find new objects, and update tracker with new objects

            new_obj_mask = segtracker.find_new_objs(track_mask,seg_mask)

            save_prediction(new_obj_mask,output_dir,str(frame_idx)+'_new.png')

            pred_mask = track_mask + new_obj_mask

            # segtracker.restart_tracker()

            segtracker.add_reference(frame, pred_mask)

        else:

            pred_mask = segtracker.track(frame,update_memory=True)

        torch.cuda.empty_cache()

        gc.collect()

        save_prediction(pred_mask,output_dir,str(frame_idx)+'.png')

        # masked_frame = draw_mask(frame,pred_mask)

        # masked_pred_list.append(masked_frame)

        # plt.imshow(masked_frame)

        # plt.show() 

        

        pred_list.append(pred_mask)

        

        

        print("processed frame {}, obj_num {}".format(frame_idx,segtracker.get_obj_num()),end='\r')

        frame_idx += 1

    cap.release()

    print('\nfinished')

# draw pred mask on frame and save as a video

cap = cv2.VideoCapture(io_args['input_video'])

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if io_args['input_video'][-3:]=='mp4':

    fourcc =  cv2.VideoWriter_fourcc(*"mp4v")

elif io_args['input_video'][-3:] == 'avi':

    fourcc =  cv2.VideoWriter_fourcc(*"MJPG")

    # fourcc = cv2.VideoWriter_fourcc(*"XVID")

else:

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))

frame_idx = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:

        break

    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    pred_mask = pred_list[frame_idx]

    masked_frame = draw_mask(frame,pred_mask)

    # masked_frame = masked_pred_list[frame_idx]

    masked_frame = cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)

    out.write(masked_frame)

    print('frame {} writed'.format(frame_idx),end='\r')

    frame_idx += 1

out.release()

cap.release()

print("\n{} saved".format(io_args['output_video']))

print('\nfinished')

# save colorized masks as a gif

imageio.mimsave(io_args['output_gif'],pred_list,fps=fps)

print("{} saved".format(io_args['output_gif']))

# manually release memory (after cuda out of memory)

del segtracker

torch.cuda.empty_cache()

gc.collect()

