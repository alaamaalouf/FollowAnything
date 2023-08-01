# --------------------------------------------------------
# FAM
# Licensed under The MIT License
# Written by Alaa Maalouf (alaam@mit.edu)
# --------------------------------------------------------
import sys
import glob
import cv2
import torch

#from segment_anything.sam_wrapper import *
#from torchvision import transforms

sys.path.append("./Segment-and-Track-Anything")
sys.path.append("./Segment-and-Track-Anything/aot")

from DRONE.drone_controller import *
from VIDEO.video import *

from collections import OrderedDict
import copy
import threading
import time
from PIL import Image
from scipy.signal import butter, filtfilt

import open_clip
from model_args import aot_args,sam_args,segtracker_args
from PIL import Image

from DINO.collect_dino_features import *
from DINO.dino_wrapper import *
from sam.segment_anything import sam_model_registry, SamPredictor
from SegTracker import SegTracker


import asyncio
import argparse
import matplotlib
import gc 
import queue


#clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
#tokenizer = open_clip.get_tokenizer('ViT-B-32')
#clip = clip.cuda()

parser = argparse.ArgumentParser(description='PyTorch + mavsdk -- zero shot detection, tracking, and drone control')

parser.add_argument('--siam_tracker_model', 
                     type=str, metavar='PATH',default= 'SiamMask/experiments/siammask_sharp/SiamMask_DAVIS.pth',
                     help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='SiamMask/experiments/siammask_sharp/config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
parser.add_argument('--cpu', action='store_true', default =False,  help='cpu mode')
parser.add_argument('--use_16bit', action='store_true',  default =False, help='16 bit dino mode')
parser.add_argument('--use_filter', action='store_true',  default =False, help='use_filter')
#parser.add_argument('--use_random_features', action='store_true',  default =False, help='')
parser.add_argument('--plot_visualizations', action='store_true', default =False, help='plot_visualizations')
parser.add_argument('--use_traced_model', action='store_true',  default =False, help='apply torch tracing')
parser.add_argument('--dino_strides', default=4, type=int , help='Strides for dino')
parser.add_argument('--desired_feature', default=[],  action='append', help='The feature we wish todetect and track from the annotated feature')
parser.add_argument('--desired_height', default=240, type=int, help='desired_height resulution')
parser.add_argument('--desired_width', default=320, type=int, help='desired_width resulution')
parser.add_argument('--queries_dir', default='./queries', help='The directory to collect the queries from')
parser.add_argument('--path_to_video', default='video/whales.mp4', help='The path to the video file')
parser.add_argument('--save_images_to', default=False, help='The path to save all semgentation/tracking frames')

parser.add_argument('--video_order', default='any', help='')
parser.add_argument('--class_threshold', default=0.7, help='Threshold below which similarity scores are assigned as not the same class')
parser.add_argument('--similarity_thresh', default=0.1, help='Threshold below which similarity scores are to be set to zero')
parser.add_argument('--min_area_size', default=100, help='')
parser.add_argument('--metric',  default='closest_mean', help='Not suppoerted on all mode, leave as default')



parser.add_argument('--detect_only', default = False, action='store_true', help='')
parser.add_argument('--use_sam', default = False, action='store_true', help='use sam')
parser.add_argument('--fps', default = 0, type=float, help='parse video frames as in fps>1')

parser.add_argument('--tracker', default='aot',help='siammask/aot')
parser.add_argument('--detect', default='dino', help='dino/click/box/clip')
parser.add_argument('--redetect_by', default='tracker', help='dino/click/box/clip/tracker')

parser.add_argument('--wait_key', default=30, type=int, help='cv waitkey')

#parser.add_argument('--drone_task', default='follow',help='land/follow')
parser.add_argument('--fly_drone', default = False, action='store_true', help='actual drone -- not simulation')
parser.add_argument('--fly_meters', default=0, type=int, help='meter to fly at the begening. If 0 then drone should be already flying')
parser.add_argument('--fly_mode', default='local', help='local/global')
parser.add_argument('--port', default='ttyUSB0', help='used port for connecting to the drone')
parser.add_argument('--baud', default='57600', help='baud rate')
parser.add_argument('--speed', default=1, type=float, help='speed m/s')
parser.add_argument('--use_yaw', default = False, action='store_true', help='use yaw')

parser.add_argument('--text_query', default='', help='')



parser.add_argument('--dont_allow_contours_mix', default = False, action='store_true', help='dont allow contours mix')
parser.add_argument('--num_of_clicks_for_detection', default=3, type = float,  help='pred_iou_thresh for sam')
parser.add_argument('--sort_by', default="area",  help='stability_score|area|predicted_iou')


args = parser.parse_args()
cmap = matplotlib.cm.get_cmap("jet")

if args.detect == 'clip':
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    clip = clip.cuda()
elif  args.detect == 'dino':
    pass
if args.tracker == 'siammask':
    sys.path.append("./SiamMask")
    sys.path.append("./SiamMask/experiments/siammask_sharp")
    from SiamMask.tools.test import *

def multiclass_vis(class_labels, img_to_viz, num_of_labels, np_used = False,alpha = 0.5):
    _overlay = img_to_viz.astype(float) / 255.0
    if np_used:
         viz = cmap(class_labels/num_of_labels)[..., :3]
    else:
         class_labels = class_labels.detach().cpu().numpy().astype(float)
         viz = cmap((class_labels/num_of_labels))[..., :3]
    _overlay =  alpha * viz + (1-alpha) * _overlay 
    s_overlay = cv2.cvtColor(np.float32(_overlay), cv2.COLOR_BGR2RGB)  

    return _overlay


def bool_mask_to_integer(mask):
    mask_obj = mask[0]
    img = np.zeros((mask_obj.shape[0], mask_obj.shape[1]))
    img[mask_obj] = 1
    return img

def get_vis_anns(anns,img_to_viz):
   
    count = 1
    dum = anns[0]['segmentation']
    img = np.zeros((dum.shape[0], dum.shape[1]))
    for ann in anns:
        m = ann['segmentation']
        img[m] = count
        count+=1
    _overlay = multiclass_vis(img, img_to_viz, count, np_used = True)
    return _overlay
    
def get_queries(cfg):
    queries = OrderedDict({})
    if cfg['detect'] == 'clip':
        input_queries = cfg['text_query']
        text = tokenizer(input_queries.split(","))
        
        text_features = clip.encode_text(text.cuda())
        text_features /= text_features.norm(dim=-1, keepdim=True)

        for idx,query in enumerate(input_queries.split(",")):
            queries[query] = text_features[idx]
        return queries

    else:
        queries = OrderedDict({})
        for file_name in os.listdir(cfg['queries_dir']):
            if file_name.startswith("feat") and file_name.endswith(".pt"):
                full_path = "{}/{}".format(cfg['queries_dir'], file_name)
                query = torch.load(full_path) 
                if not isinstance(query, list): # annotations
                    query = [query]
                key = file_name[4:-3]
                queries[key] = query
            
        if not queries.keys(): 
                print("No annotations found in {}!!!!, see step 1 and script annotate_features.py".format(cfg['queries_dir']))
                exit("1")

        if cfg['metric'] == 'closest_mean':
            mean_queries = OrderedDict({})
            for key,query in queries.items():
                query = torch.stack(query).cuda().mean(dim=0)
                query = torch.nn.functional.normalize(query, dim=0)
                mean_queries[key] = query
            return mean_queries
        else:
            return queries


def get_aot_tracker_with_sam():
    ###modify args if needed###
    segtracker = SegTracker(segtracker_args, sam_args, aot_args)
    segtracker.restart_tracker()
    return segtracker

def get_siammask_tracker(siam_cfg, device):

    from custom import Custom
    
    siammask = Custom(anchors=siam_cfg['anchors'])
    if args.siam_tracker_model:
        assert isfile(args.siam_tracker_model), 'Please download {} first.'.format(args.siam_tracker_model)
        siammask = load_pretrain(siammask, args.siam_tracker_model)
    siammask.eval().to(device)

    return siammask

def plot_similarity_if_neded(cfg, frame, similarity_rel, alpha = 0.5):
    if cfg['plot visualizations'] or cfg["save_images_to"]:
        img_to_viz = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_to_viz = cv2.resize(img_to_viz, (similarity_rel.shape[-1], similarity_rel.shape[-2]))
        similarity_colormap = cmap(similarity_rel)[..., :3]
        
        _overlay = img_to_viz.astype(np.float32) / 255
        _overlay = (1-alpha) * _overlay + (alpha) * similarity_colormap
        _overlay = cv2.cvtColor(np.float32(_overlay), cv2.COLOR_BGR2RGB)
        plot_and_save_if_neded(cfg, _overlay, "DINO-CLIP-result")
    
            
def plot_and_save_if_neded(cfg, image_to_plot, stage_and_task, count, multiply = 1):
    global mission_counter

    if cfg['plot_visualizations']: 
        cv2.imshow(stage_and_task, image_to_plot)
        cv2.waitKey(cfg['wait_key'])
    if cfg['save_images_to']:
        file_name = "{}/{}/{}_{}.jpg".format(cfg['save_images_to'],stage_and_task,mission_counter ,count)
        #if os.path.exists(filename):
        cv2.imwrite(file_name,image_to_plot*multiply)

def get_dino_result_if_needed(cfg, frame, class_labels):
        #if cfg['plot visualizations'] or cfg["save_images_to"]:
        _overlay = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _overlay = cv2.resize(_overlay, (cfg['desired_width'],cfg['desired_height']))
        _overlay = multiclass_vis(class_labels, _overlay, len(queries))
        return _overlay
        #plot_and_save_if_neded(cfg, _overlay, "DINO-result-only")

def automatic_object_detection(vit_model, sam, video, queries, cfg, vehicle):
    count=0; detecton_count = 0
    detected = False
    cosine_similarity = torch.nn.CosineSimilarity(dim=1)  # (1, 512, H // 2, W // 2)
    with torch.cuda.amp.autocast():
        while not detected: #!and video.isOpened():
            s = time.time()
            read_one_frame = False
            while not read_one_frame:
                read_one_frame, frame = video.read()

            
            frameshow = cv2.resize(frame, (cfg['desired_width'],cfg['desired_height']))
            saved_frame = copy.copy(frameshow)   
            plot_and_save_if_neded(cfg, frameshow, 'Stream_segmenting', count) 
          
            if cfg['use_sam']:
                t = time.time()
                _, masks = sam.seg(frameshow)
                print("Sam took: ", time.time() - t)
                masks = sorted(masks, key=(lambda x: x[cfg['sort_by']]), reverse=True)
                print("Sam generated {} masks".format(len(masks)))
                masks_of_sam = get_vis_anns(masks, frameshow)
                plot_and_save_if_neded(cfg, masks_of_sam, 'SAM-result', count,multiply = 255)
                
            
            t = time.time()
            if cfg['detect'] == "dino":
                frame_preprocessed = preprocess_frame(frame, cfg=cfg)
                img_feat = vit_model.forward(frame_preprocessed)
                img_feat_norm = torch.nn.functional.normalize(img_feat, dim=1)
                #print("Dino took", time.time() - t)

            #smoothing by mean
            if cfg['use_sam']:
                cosine_similarity = torch.nn.CosineSimilarity()
                class_labels = torch.zeros((frameshow.shape[0],frameshow.shape[1]))
                thresh = torch.zeros((frameshow.shape[0],frameshow.shape[1]))
                all_masks_sims = []
                for ii,mask in enumerate(masks):
                    mask_similarity = []
                    m = mask['segmentation']
                    if  cfg['detect'] == 'clip':
                        #if ii == 0: continue
                        _x, _y, _w, _h = tuple(mask["bbox"])  # xywh bounding box
                        
                        img_roi = frameshow[_y : _y + _h, _x : _x + _w, :]
                        
                        img_roi = Image.fromarray(img_roi)
                        img_roi = clip_preprocess(img_roi).unsqueeze(0).cuda()
                        roifeat = vit_model.encode_image(img_roi)

                        mask_feature = torch.nn.functional.normalize(roifeat, dim=-1)
                    else:
                        mask_feature = img_feat_norm[:,:,m].mean(axis=2)
                    
                    tmp_map_dict = {}
                    counter_item = 0
                    for _idx, query in queries.items():
                        mask_similarity.append(cosine_similarity(mask_feature.reshape(1,-1), query.reshape(1,-1)))
                        tmp_map_dict[counter_item]  = _idx
                        counter_item += 1
                   
                    if len(queries.items()) >1 :
                        if  cfg['detect'] == 'clip' and ii ==0: 
                            continue
                        mask_label = torch.argmax(torch.as_tensor(mask_similarity))  # 1, H, W 
                        
                        mask['label'] = tmp_map_dict[int(mask_label)]
                        #print(tmp_map_dict[int(mask_label)])
                        #print(m)
                        #pred_mask = bool_mask_to_integer(m)
                        #vis_mask = multiclass_vis(m, frameshow, 2, np_used = True)
                        #plot_and_save_if_neded(cfg, vis_mask, 'Tracker-result',count,multiply = 255)
                        #input()
                        class_labels[m] = int(mask_label) + 1
                        if mask['label'] in cfg['desired_feature']:
                            thresh[m] = 1 
                    else:
                        all_masks_sims.append(mask_similarity[0]) 
                        if float(mask_similarity[0]) > float(cfg['class_threshold']):  thresh[m] = 1 
                #if  len(queries.items()) == 1 and cfg['query_type'] == 'text':      #if mask_similarity[0] > 0.9:
                #    sorted_sims = np.argsort(all_masks_sims)
                #    for masks in masks[sorted_sims[:k]]
                #    thresh[m] = 1 
                #    mask['dino_label'] = 1
                        
            else:
                if len(queries.keys()) == 1:
                    query = queries[queries.keys()[0]].cuda()
                    similarity = cosine_similarity(img_feat_norm, query.view(1, -1, 1, 1))
                    similarity = (similarity + 1.0) / 2.0  # scale from [-1, 1] to [0, 1]
                    similarity_rel = (similarity - similarity.min()) / (similarity.max() - similarity.min() + 1e-12)
                    similarity_rel = similarity_rel[0]  # 1, H // 2, W // 2 -> # H // 2, W // 2
                    similarity_rel[similarity_rel < cfg['similarity_thresh']] = 0.0
                    similarity_rel = similarity_rel.detach().cpu().numpy()
                    plot_similarity_if_neded(cfg, frame, similarity_rel, alpha = 0.5)
                    ret, thresh = cv2.threshold(similarity_rel*255, cfg['class_threshold'], 255, 0)          
                else:
                    similarities = []
                    tmp_map_dict = {}
                    counter_item = 0
                    t = time.time()
                    for _idx, query in queries.items():
                        if cfg['metric'] == 'closest_mean':
                            similarity = cosine_similarity(img_feat_norm, query.view(1, -1, 1, 1))
                            similarities.append(similarity)
                            
                        elif cfg['metric'] in ['closest_feature','closest_mean_of_similarity']:
                            class_similarity = []
                            for single_annotitation in query:
                                single_annotitation = torch.nn.functional.normalize(single_annotitation.cuda(), dim=0)
                                similarity = cosine_similarity(img_feat_norm, single_annotitation.view(1, -1, 1, 1))
                                class_similarity.append(similarity)
                            class_similarity=torch.stack(class_similarity)
                            if cfg['metric'] == 'closest_feature':
                                class_similarity = torch.max(class_similarity, dim=0)[0]
                            else:
                                class_similarity = torch.mean(class_similarity, dim=0)
                            similarities.append(class_similarity)
                        tmp_map_dict[_idx] = counter_item
                        counter_item+=1
                    similarities = torch.stack(similarities)
                    class_labels = torch.argmax(similarities, dim=0)  # 1, H, W
                    class_labels = class_labels[0]

                    thresh = copy.deepcopy(class_labels)
                    for desired_feat in cfg['desired_feature']:
                        feat = tmp_map_dict[desired_feat]
                        thresh[thresh == feat] = 255
                    thresh[thresh != 255] = 0
                    
            
            if cfg['plot_visualizations'] or cfg['save_images_to']:
                    dino_plot = get_dino_result_if_needed(cfg, frame, class_labels) 
            

            detections = [] 
            all_masks = np.zeros((thresh.shape[0],thresh.shape[1]))
            if not cfg['dont_allow_contours_mix'] or not cfg['use_sam']:  
                thresh = thresh.cpu().detach().numpy().astype("uint8")
                marker_count, contours = cv2.connectedComponents(thresh)
               
                for label_for_detected_obj in range(1,marker_count):
                    bool_mask = contours==label_for_detected_obj
                    mask = np.where(bool_mask, np.uint8(255), np.uint8(0))
                    x,y,w,h = cv2.boundingRect(mask)
                    area = cv2.countNonZero(mask[y:y+h,x:x+w])
                    
                    if area > cfg['min_area_size']:
                        detections.append([x, y, w, h])
                        all_masks[bool_mask] = label_for_detected_obj
                        if cfg['plot_visualizations'] or cfg['save_images_to']:
                            cv2.rectangle(dino_plot,(x,y),(x+w,y+h),(255,0,0),3)
                            cv2.rectangle(frameshow,(x,y),(x+w,y+h),(255,0,0),3)
                   
                    if not mask.any(): break
                    
            else: 
                label_for_detected_obj = 1
                for n, mask in enumerate(masks):
                    m = mask['segmentation']
                    
                    if (thresh[m] == 1).all():
                        all_masks[m] = label_for_detected_obj
                        label_for_detected_obj +=1
                        x, y, w, h = mask['bbox']
                    
                        detections.append([x, y, w, h])
                        if cfg['plot_visualizations'] or cfg['save_images_to']:
                            cv2.rectangle(dino_plot,(x,y),(x+w,y+h),(255,0,0),3)
                            cv2.rectangle(frameshow,(x,y),(x+w,y+h),(255,0,0),3)
                        
                     
            count +=1
            if len(detections)>0:
                plot_and_save_if_neded(cfg, dino_plot, "DINO-CLIP-result",count, multiply =255)
                plot_and_save_if_neded(cfg, frameshow, "Detection",count)
                #plot_and_save_if_neded(cfg, all_masks*255, "all_masks") 
                detecton_count+=1
                if detecton_count == 1 and not cfg['detect_only']:
                        print("Found {} desired objects!".format(len(detections)))
                        print("Moving to tracking")
                        return detections, all_masks.astype(float), saved_frame
            torch.cuda.empty_cache()
            gc.collect()

            #drone_action_wrapper_while_detecting(vehicle,cfg)
            print("Time took: ",time.time()-s)  


def compute_area_and_center(bounding_shape):
    x = 0; y = 1
    bounding_shape = bounding_shape[0]#(bounding_shape)
    idx_right = np.argmax(bounding_shape[:,x]) 
    idx_left = np.argmin(bounding_shape[:,x]) 
    idx_up = np.argmax(bounding_shape[:,y]) 
    idx_bottum = np.argmin(bounding_shape[:,y]) 


    right_point = bounding_shape[idx_right]
    left_point = bounding_shape[idx_left]
    up_point = bounding_shape[idx_up]
    bottum_point = bounding_shape[idx_bottum]


    #area, center = compute_area_and_center(right_point, left_point, up_point, bottum_point)
    area = np.linalg.norm(up_point - left_point)* np.linalg.norm(up_point - right_point)
    center = np.mean([right_point, left_point, up_point, bottum_point], axis = 0)
   
    return area, center

def track_object_with_siammask(siammask, detections, video, cfg, tracker_cfg, vehicle):
    x, y, w, h = detections[0]#todo
    print(x, y, w, h)
    toc = 0
    f=0
    while 1:

        ret, im = video.read()
        if not ret:
            print("No stream!!!")
            break 
        im = cv2.resize(im, (cfg['desired_width'],cfg['desired_height']))
        im_store = copy.deepcopy(im)
        tic = cv2.getTickCount()
        
        if f == 0:  # init
            
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, tracker_cfg['hp'], device=device)  # init tracker
        elif f > 0:  # tracking
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr

            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]

            bounding_shape = np.int0(location).reshape((-1, 1, 2))
            _, mean_point = compute_area_and_center(bounding_shape)
            compute_drone_action_while_tracking(mean_point, cfg, vehicle)

            cv2.polylines(im, [bounding_shape], True, (0, 255, 0), 3)
            cv2.imshow('Tracker-result', im)
            if cfg['save_images_to']:
                cv2.imwrite("{}/Tracker-result/{}.jpg".format(cfg['save_images_to'],f),im) 
            key = cv2.waitKey(cfg['wait_key'])
            if key > 0: break
            if cfg['save_images_to']:
                cv2.imwrite("{}/Stream_tracking/{}.jpg".format(cfg['save_images_to'],f),im_store) 

        f+=1
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))


def create_video_from_images(cfg):

    import glob
    vfile= '{}/video_from_images.avi'.format(cfg['path_to_video'])
    fileidx = 0
    if not  os.path.exists(vfile):
        img_array = []
        if cfg['video_order'] == 'any':
            for filename in os.listdir(cfg['path_to_video']):
                filename = os.path.join(cfg['path_to_video'],filename )
                img = cv2.imread(filename)
                height, width, layers = img.shape
                size = (width,height)
                img_array.append(img)
                fileidx+=1
        else:
            while 1:
                filename = os.path.join(cfg['path_to_video'], f"{fileidx:06d}.png")
                if not os.path.exists(filename): 
                    filename = os.path.join(cfg['path_to_video'], "1_{}.jpg".format(fileidx))#f"{fileidx}.jpg")
                if not os.path.exists(filename):    
                    break
                img = cv2.imread(filename)
                height, width, layers = img.shape
                size = (width,height)
                img_array.append(img)
                fileidx+=1

        out = cv2.VideoWriter('{}/video_from_images.avi'.format(cfg['path_to_video']),cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
         
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
    cfg['path_to_video'] = vfile
    return cv2.VideoCapture(cfg['path_to_video']) 

def init_system():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    cfg = vars(args)
    
    # Setup Tracker Model
    print("Init tracker...")
    if cfg['tracker'] == 'siammask':
        tracker_cfg = load_config(args)
        tracker = get_siammask_tracker(siam_cfg = tracker_cfg, device = device)
    elif  cfg['tracker'] == 'aot':
        tracker = get_aot_tracker_with_sam()
        tracker_cfg = None
    else:
        print("Exiting. No such tracker {}".format(cfg['tracker']))
        exit(9)

    
    print("Init Segmentor...")
    # Setup 0-shot segmentor
    if cfg['tracker'] !=  'aot':
        segmentor = get_aot_tracker_with_sam()
    else:
        segmentor = tracker

    
    # Setup 0-shot detector
    print("Init Detector...")
    if cfg['detect'] == 'clip':
        if cfg['text_query'] == "":
            print("Exiting. No text query is provided while using clip")
            exit(9)
        queries = get_queries(cfg=cfg)
        detector = clip
        
    elif cfg['detect'] == "dino":
        detector = get_dino_pixel_wise_features_model(cfg = cfg, device = device)
        print("Init queries...")
        # Setup queries to compare
        queries = get_queries(cfg=cfg)
    elif cfg['detect'] in ["click","box"]:
        sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint']).to(device=device)
        detector = SamPredictor(sam)
        queries = None
    else:
        print("Exiting. No such detector {}".format(cfg['detect']))
        exit(9)

    
    print("Init video...")
    if os.path.isdir(cfg["path_to_video"]):
        print("Making video from images in directory {}".format(cfg["path_to_video"]))
        video = create_video_from_images(cfg)
    elif os.path.exists(cfg["path_to_video"]) and cfg['fps']<1:
        print("Reading video  {}".format(cfg["path_to_video"]))
        video = cv2.VideoCapture(cfg["path_to_video"]) 
    else:
        print("Using stream from {}".format(cfg["path_to_video"]))
        video = ThreadedCamera(cfg["path_to_video"], fps = cfg['fps'])
    

    if cfg["fly_drone"]:
        print("Init Drone...")
        vehicle = loop.run_until_complete(init_drone(port = cfg['port'], 
                                                     baud = cfg['baud'], 
                                                     fly_meters = cfg['fly_meters'],
                                                     speed = cfg['speed'],
                                                     fly_mode = cfg['fly_mode']))
    else:
        vehicle = None
   

    if cfg['save_images_to']:
        create_dir_if_doesnt_exists(cfg['save_images_to'])
        for directory_to_create in ['SAM-result', 'Stream_segmenting', 'DINO-CLIP-result', 'Tracker-result', 'Stream_tracking', 'Detection']:
            create_dir_if_doesnt_exists(os.path.join(cfg['save_images_to'],directory_to_create))

    return device, tracker_cfg, cfg, tracker, detector, segmentor, queries, video, vehicle




def create_dir_if_doesnt_exists(dir_to_create):
    if not os.path.exists(dir_to_create): os.mkdir(dir_to_create)



# Called every time a mouse event happen
def on_mouse(event, x, y, flags, userdata):
    global state, p1, p2
    # Left click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Select first point
            p1 = (x,y)
            state += 1
    elif event == cv2.EVENT_LBUTTONUP:
        # Select second point
        if state == 1:
            p2 = (x,y)
            state += 1
    # Right click (erase current ROI)
    if event == cv2.EVENT_RBUTTONUP:
        p1, p2 = None, None
        state = 0

def click_on_object(event, x, y, flags, userdata):
    global state,points
    # Left click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Select first point
        points.append([x,y])
        labels.append(1)
        state += 1
    # Right click (erase current ROI)
    if event == cv2.EVENT_RBUTTONDOWN:
        points.append([x,y])
        labels.append(0)
        state += 1

# Register the mouse callback

def detect_by_click(sam , video, cfg, vehicle):
    global state, points, labels
    points = []
    labels = []
    state = 0
    cv2.namedWindow('Choose_object')#, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Choose_object', click_on_object)    
    while 1:
        read_one_frame = False
        while not read_one_frame: #and video.isOpened():
            read_one_frame, frame = video.read()
            #print(read_one_frame)
        #print( (cfg['desired_width'],cfg['desired_height']))
        frame = cv2.resize(frame, (cfg['desired_width'],cfg['desired_height']))
        cv2.imshow('Choose_object', frame)
        key = cv2.waitKey(cfg['wait_key'])
        if key == 27: exit(9)

        if state >= cfg['num_of_clicks_for_detection']:
            sam.set_image(frame)
            input_point = np.array(points)
            input_label = np.array(labels)

            masks, scores, logits = sam.predict(
                                            point_coords=input_point,
                                            point_labels=input_label,
                                            multimask_output=False,
                                        )

            return None, masks, frame 


def compute_drone_action_while_tracking(mean_point, cfg, vehicle):
    global mean
    global postion_vector_queue
    x = 1
    y = 0
    z = 2
    
    heading = 0
    if mean_point is None:
        postion_vector_queue.queue.clear()
        print("Stopping to detect the object again")
        loop.run_until_complete(move_drone_by_velocity(vehicle, 0, 
                                                        0, 0, 
                                                        0, 
                                                        K = 0, 
                                                        yaw_K = 0))
        return
    if cfg['use_yaw']:
        y_p = abs(cfg['desired_height'])/2 - mean_point[y]
        x_p = abs(cfg['desired_width'])/2 - mean_point[x]
        heading =  (-np.arctan2(x_p,y_p))*180/np.pi

    normalized_center = np.zeros(3)
    normalized_center[x] = mean_point[x] / cfg['desired_width']  #obtain a x center between 0.0 and 1.0  
    normalized_center[y] = mean_point[y] / cfg['desired_height'] #obtain a y center between 0.0 and 1.0  
    normalized_center[z] = heading

    direction_vector = normalized_center - np.array([0.5,0.5,0])#todo
    direction_vector[y]*=-1

   
    

    if  cfg['use_filter']: postion_vector_queue.put(direction_vector)

    if postion_vector_queue.full():
        p_filt = filtfilt(cof_b, cof_a,  
                        np.array(list(postion_vector_queue.queue)),
                        axis=0)
        postion_vector_queue.get()
        loop.run_until_complete(move_drone_by_velocity(vehicle, p_filt[-1][0], 
                                                        p_filt[-1][1], 0, 
                                                        p_filt[-1][2], 
                                                        K = 1.5, yaw_K = 0.15 ))
    else:
        loop.run_until_complete(move_drone_by_velocity(vehicle, direction_vector[0], 
                                                       direction_vector[1], 0, 
                                                       direction_vector[2],
                                                       K= 1.5, yaw_K = 0.15 ))
    
def track_object_with_aot(tracker, pred_mask, frame,  video, cfg, vehicle, track_single_object = True):
    
    tracker.restart_tracker()

    if track_single_object:
        pred_mask[pred_mask!=1] = 0 
    
    mean_points  = []
    timing = 0; frame_idx = 0
    
    with torch.cuda.amp.autocast():
        while 1: # 
            t = time.time()
            ##############################################
            #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            if frame_idx == 0:
                torch.cuda.empty_cache()
                gc.collect()
                tracker.add_reference(frame, pred_mask)
            else:
                pred_mask = tracker.track(frame,update_memory=True)

            if track_single_object: pred_mask[pred_mask!=1] = 0 
            torch.cuda.empty_cache()
            gc.collect()
            ###############################################

            mean_point = get_mean_point(pred_mask)
            if mean_point is None and cfg['redetect_by']!= 'tracker': return "FAILED"
            
            compute_drone_action_while_tracking(mean_point, cfg, vehicle)
           
            ##############################################
            vis_masks = multiclass_vis(pred_mask, frame, np.max(pred_mask) + 1, np_used = True)
            plot_and_save_if_neded(cfg, frame, "Stream_tracking",frame_idx)
            plot_and_save_if_neded(cfg, vis_masks, 'Tracker-result',frame_idx,multiply = 255)
            print("processed frame {}, obj_num {}".format(frame_idx,tracker.get_obj_num()),end='\r')
            
            ##############################################

            frame_idx += 1
            read_one_frame = False
            while not read_one_frame:
                read_one_frame, frame = video.read()

            frame = cv2.resize(frame, (cfg['desired_width'],cfg['desired_height']))
        
def get_mean_point(pred_mask, bounding_shape = None):
    
    if not pred_mask is None:
        object_indx = (pred_mask == 1).nonzero()

        if object_indx[0].shape[0] == 0:
            return None ## restart mission
    mean_point = [int(object_indx[0].mean()),  int(object_indx[1].mean())]
    return mean_point
def detect_by_box(sam , video, cfg, vehicle): 
    cv2.namedWindow('Choose_object')#, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Choose_object', on_mouse)
    # Our ROI, defined by two points
    global state, p1, p2
    p1, p2 = None, None
    state = 0
    
    while 1:
        read_one_frame = False
        while not read_one_frame:
            read_one_frame, frame = video.read()

        frame = cv2.resize(frame, (cfg['desired_width'],cfg['desired_height']))
        # If a ROI is selected, draw it
        if state > 1:
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 5)
        # Show image
        cv2.cvtColor(np.float32(frame), cv2.COLOR_BGR2RGB)
        cv2.imshow('Choose_object', frame)
        # Let OpenCV manage window events
        key = cv2.waitKey(cfg['wait_key'])
        # If ESCAPE key pressed, stop
        if key == 27: video.release()
        

        if state > 1:
            if cfg['tracker'] == 'siammask':
                res = [[p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1] ]]
                return res, None, frame
            else:
                sam.set_image(frame)
                input_box = np.array([p1[0], p1[1], p2[0], p2[1]])
                masks, _, _ = sam.predict(
                                #point_coords=input_point,
                                #point_labels=input_label,
                                box=input_box,
                                multimask_output=False,
                        )

                return input_box, masks, frame
 
        #drone_action_wrapper_while_detecting(vehicle,cfg)

def detect_object(cfg, detector, segmentor, video, queries):
    print("aplying {} detection...".format(cfg['detect']))
    if cfg['detect'] in ['dino', 'clip']:
        bounding_boxes, masks_of_sam, saved_frame = automatic_object_detection(vit_model=detector, sam = segmentor, 
                                                                               video=video, queries=queries, 
                                                                               cfg=cfg, vehicle=vehicle)
        vis_masks = multiclass_vis(masks_of_sam, saved_frame, np.max(masks_of_sam) + 1, np_used = True, alpha =1)
        #plot_and_save_if_neded(cfg, vis_masks, 'Tracker-result')
    else:
        if cfg['detect'] == 'click':
            bounding_boxes, masks, saved_frame = detect_by_click(sam = detector, video=video, cfg=cfg, vehicle=vehicle)
        
        else:
            bounding_boxes, masks, saved_frame = detect_by_box(sam = detector, video=video, cfg=cfg, vehicle=vehicle)
        
        if  masks is not None:
            masks_of_sam = bool_mask_to_integer(masks)
            vis_masks = multiclass_vis(masks_of_sam, saved_frame, 2, np_used = True)
            plot_and_save_if_neded(cfg, vis_masks, 'Choose_object',0)
        
        else:
            masks_of_sam = None 
    return bounding_boxes, masks_of_sam, saved_frame
def start_mission(device, tracker_cfg, cfg, tracker, detector, segmentor, queries, video, vehicle):
    global mission_counter
    mission_counter +=1

    if mission_counter > 1 and cfg['redetect_by'] in ['dino','clip']:
        cfg['detect'] = cfg['redetect_by']

    bounding_boxes, masks, saved_frame = detect_object(cfg, detector, segmentor, video, queries)
    
    if cfg['tracker'] == "siammask":
        track_object_with_siammask(siammask=tracker, 
                                   detections=bounding_boxes, 
                                   video=video, cfg=cfg, 
                                   tracker_cfg =tracker_cfg, 
                                   vehicle = vehicle)
    else:
        status = track_object_with_aot(tracker, masks, saved_frame,  
                                       video, cfg, vehicle)
        if status == 'FAILED': 
            print("Redtecting....")
            start_mission(device, tracker_cfg, cfg, tracker, 
                          detector, segmentor, queries, 
                          video, vehicle)
if __name__ == '__main__':
    # Setup device

    global mission_counter 
    mission_counter = 0
    loop = asyncio.get_event_loop()
    global postion_vector_queue
    postion_vector_queue = queue.Queue(30) #The max size is 5.
    FREQ = 0.5
    cof_b, cof_a = butter(6, FREQ, fs=17, btype='low')

    device, tracker_cfg, cfg, tracker, detector, segmentor, queries, video, vehicle = init_system()
    start_mission(device, tracker_cfg, cfg, tracker, detector, segmentor, queries, video, vehicle )
    

 