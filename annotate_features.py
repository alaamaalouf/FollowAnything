import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
import sys


import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
#from lseg import LSegNet

import warnings
from pathlib import Path
from typing import Union

import torch
import torchvision
from typing_extensions import Literal
import time
import argparse
from DINO.collect_dino_features import *
from DINO.dino_wrapper import *

parser = argparse.ArgumentParser(description='Annotation script')
parser.add_argument('--cpu', action='store_true', default =False,  help='cpu mode')
parser.add_argument('--use_16bit', action='store_true',  default =False, help='16 bit dino mode')
parser.add_argument('--plot_similarity', action='store_true', default =False, help='16 bit dino mode')
parser.add_argument('--use_traced_model', action='store_true',  default =False, help='apply torch tracing')
parser.add_argument('--dino_strides', default=4, type=int , help='Strides for dino')
parser.add_argument('--desired_height', default=240, type=int, help='desired_height resulution')
parser.add_argument('--desired_width', default=320, type=int, help='desired_width resulution')
parser.add_argument('--queries_dir', default='./queries/whales_detection', help='The directory to collect the queries from')
parser.add_argument('--similarity_thresh', default=0.1, help='Threshold below which similarity scores are to be set to zero')
parser.add_argument('--path_to_images', default='./frames', help='The directory to collect the images from')

args = parser.parse_args()

global clickx, clicky


def onclick(click):
    global clickx, clicky
    clickx = click.xdata
    clicky = click.ydata
    plt.close("all")


if __name__ == "__main__":



    # Process command-line args (if any)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    cfg = vars(args)

    if not os.path.exists(cfg['queries_dir']):
        os.mkdir(cfg['queries_dir'])

    model = get_dino_pixel_wise_features_model(cfg = cfg, device = device)


    #traced_16 = torch.jit.trace(model, torch.randn(img.shape).half().to("cuda"))

   
    # Image file indx to annotate
    #inds_to_annotate = [1, 15, 783, 1839, 1964, 2144, 2262, 8011, 8234]  # for wather, head, body, fluke
    # inds_to_annotate = [49, 59, 69, 70, 195, 205, 215, 525, 535, 545, 550]  # for spout
    #inds_to_annotate = [1, 1964, 2144, 2262, 8011, 8234]
    annotated_feats = []
    annot_classes = []

    
  
    with torch.no_grad():
    
        for imgname in os.listdir(cfg['path_to_images']):

            imgfile = os.path.join(cfg['path_to_images'], imgname)
            img = cv2.imread(imgfile)
            img = preprocess_frame(img, cfg = cfg)
            

            img_feat = model(img)
           
            img_feat_norm = torch.nn.functional.normalize(img_feat, dim=1)

            img_to_viz = cv2.imread(imgfile)
            img_to_viz = cv2.cvtColor(img_to_viz, cv2.COLOR_BGR2RGB)
            img_to_viz = cv2.resize(img_to_viz, (img_feat_norm.shape[-1], 
                                                img_feat_norm.shape[-2])
                                    )

            while True:

                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.imshow(img_to_viz)
                plt.title("Click point to compute feature similarity wrt")
                cid = fig.canvas.mpl_connect("button_press_event", onclick)
                plt.show()

                # Compute cosine similarity between feature vector of selected point
                # and all other features in the image
                clickx = int(math.floor(clickx))
                clicky = int(math.floor(clicky))
                cosine_similarity = torch.nn.CosineSimilarity(dim=1)  # (1, 512, H // 2, W // 2)
                selected_embedding = img_feat_norm[0, :, clicky, clickx]  # (512,)
                similarity = cosine_similarity(
                    img_feat_norm, selected_embedding.view(1, -1, 1, 1)
                )
                # Viz thresholded "relative" attention scores
                similarity = (similarity + 1.0) / 2.0  # scale from [-1, 1] to [0, 1]
                # similarity = similarity.clamp(0., 1.)
                similarity_rel = (similarity - similarity.min()) / (
                    similarity.max() - similarity.min() + 1e-12
                )
                similarity_rel = similarity_rel[0]  # 1, H // 2, W // 2 -> # H // 2, W // 2
                similarity_rel[similarity_rel < args.similarity_thresh] = 0.0
                cmap = matplotlib.cm.get_cmap("jet")
                similarity_colormap = cmap(similarity_rel.detach().cpu().numpy())[..., :3]

                _overlay = img_to_viz.astype(np.float32) / 255
                _overlay = 0.5 * _overlay + 0.5 * similarity_colormap
                plt.imshow(_overlay)
                plt.scatter([clickx], [clicky])
                plt.show()

                plt.close("all")

                class_label = input("Enter class label ('s' to discard): ")
                if class_label != "s":
                    try:
                        annot_classes.append(int(class_label))
                    except:
                        print("labels are only allowed to be integers, {} is not".format(class_label))
                    annotated_feats.append(selected_embedding.detach().cpu())
                # Prompt user whether or not to continue
                

                # water, head, body, fluke
                #feat_0, feat_1, feat_2, feat_3 = [], [], [], []
                # feat_4 = []  # Spout
                dict_of_annotations = {}
                for idx, cid in enumerate(annot_classes):
                    if not cid in dict_of_annotations.keys():
                        dict_of_annotations[cid] = []
                    dict_of_annotations[cid].append(annotated_feats[idx])
                   

                for key,_feat in dict_of_annotations.items(): #, feat_4]:
                    if len(_feat) == 0:
                        continue
                    #_feat = torch.stack(_feat)
                    #_feat = _feat.mean(0)
                    #_feat = torch.nn.functional.normalize(_feat, dim=0)

                    savefile = os.path.join(cfg['queries_dir'],"feat{}.pt".format(key))
                    torch.save(_feat, savefile)
               
                to_continue = input("Click another point? ('q' to quit): ")
                if to_continue == "q":
                    break
                # savefile = "feat4.pt"
                # torch.save(feat_4, savefile)