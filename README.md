# Follow Anything

**[1 CSAIL, MIT](https://www.csail.mit.edu/)**  | **[2 SEAS, Harvard](https://www.csail.mit.edu/](https://seas.harvard.edu/)https://seas.harvard.edu/)**

*[Alaa Maalouf](https://www.csail.mit.edu/person/alaa-maalouf), [Ninad Jadhav](https://react.seas.harvard.edu/people/ninad-jadhav), [Krishna Murthy Jatavallabhula](https://krrish94.github.io/), [Makram Chahine](https://www.mit.edu/~chahine/), [Daniel  M.Vogt](https://www.danielmvogt.com/), [Robert J. Wood](https://wyss.harvard.edu/team/associate-faculty/robert-wood/), [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/), and [Daniela Rus](https://danielarus.csail.mit.edu/)*

![FAn design](Images_and_videos_for_Github_visualizations/teaser.png?raw=true)



*FAn* - Follow Anything is a robotic system to detect, track, and follow any object in real-time while accounting for occlusion and object re-emergence. 
*FAn* is an open-vocabulary and multimodal system -- it is not restricted to concepts seen at training time and can be initialized/queried using text, images, or clicks.  

## Demo Videos

1. Car following and re-detecting: 
<p align="center">
<img src="Images_and_videos_for_Github_visualizations/Car_following.gif" width="400">
 <img src="Images_and_videos_for_Github_visualizations/Car_tracking.gif" width="200">

2.  Drone following:
<p align="center">
 <img src="Images_and_videos_for_Github_visualizations/drone_following.gif" width="400">
 <img src="Images_and_videos_for_Github_visualizations/drone_tracking.gif" width="200">
</p>

3. Manually moving brick following and re-detecting:
<p align="center">
<img src="Images_and_videos_for_Github_visualizations/Brick_following.gif" width="400">
 <img src="Images_and_videos_for_Github_visualizations/Brick_tracking.gif" width="200">
</p>





## Installation
The code was tested with `python=3.9.12`, as well as `pytorch=1.9.0+cu102` and `torchvision=0.10.0+cu102`. 

Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

1. Clone the repository locally:

```

 git clone https://github.com/alaamaalouf/FollowAnything.git

```

 2. Install the directory Segment-and-Track-Anything as explained in: https://github.com/z-x-yang/Segment-and-Track-Anything

    Download:

        2.1. SAM model to Segment-and-Track-Anything/ckpt directory, the default model is SAM-VIT-B (sam_vit_b_01ec64.pth).
    
        2.2. DeAOT/AOT model to Segment-and-Track-Anything/ckpt directory, the default model is R50-DeAOT-L (R50_DeAOTL_PRE_YTB_DAV.pth).


    Note - some files are slightly modified in the directory Segment-and-Track-Anything, hence, use the version provided in this directory.
    

 4. If you wish to use SiamMask as a tracker (default is AOT from step "2") install SiamMask as detailed in: https://github.com/foolwood/SiamMask

     Note - some files are slightly modified in the directory Segment-and-Track-Anything, hence, use the version provided in this directory.

 5. pip install mavsdk (you may need to do more simple pip installs for other libraries).


## Example usage on offline video
First, we show how to run the system, without the drone or the online stream, i.e., we show how to detect and track a desired object from a video.

1. To manually detect and track a desired object by a bounding box: 
```
python follow_anything.py  --desired_height 240 --desired_width 320 --path_to_video <PATH TO VIDEO> --save_images_to outputs/ --detect box --redetect_by box --tracker aot --plot_visualizations
```
        a. --desired_height and desired_height: the desired image width and height to work with.

        b. --path_to_video: full or relative path to the desired video.

        c. --save_images_to: will store all outputs of "segmentations/tracking/detection" to the provided directory.

        d. --detect: either by box, click, dino, or clip -- dino and clip require additional flags (see next section).

        e. --tracker: either aot or SiamMask

        f. --plot_visualizations: plots all stages visualizations. 

2.   To automatically detect and track a desired object we apply the following two stages:

     2.1 annotation phase - run:

     
     ```
     python annotate_features.py --desired_height 240 --desired_width 320 --queries_dir <directory where to store the queries features> --path_to_images <path to a directory containing the images we wish to annotate>
     ```
     
     2.1. Run the system with --detect dino -redetect_by dino
     ```
     python follow_anything.py  --desired_height 240 --desired_width 320 --path_to_video <PATH TO VIDEO> --save_images_to outputs/  --detect dino --redetect_by dino --use_sam --tracker aot --queries_dir <directory where you stored the queries features in step a>  --desired_feature <desired_label>  --plot_visualizations
     ```
           a. queries_dir: the directory where you stored the queries features in step a
           b. desired_feature:  the label of the desired annotated object.
           c. use_sam: use sam before detection to provide segmentation (slower but better) ---> You can remove this flag to get faster detection.

3. For faster Dino detection performance add:
``` --use_16bit --use_traced_model```  and remove ```--use_sam``` (this mode is less accurate but more efficient).
4. To use text for detection add:
```
--detect clip --desired_feature <text explaining the desired feature as well as possible>  --use_sam  --text_query <text explaining the desired feature as well as possible, text explaining object two, ..., text explaining the last object in the scene>
```     


## Example usage on a video stream and a drone
All you need is to pick the relevant command as explained in the section above and add the flags ```--path_to_video rtsp://192.168.144.10:8554/H264Video --fly_drone --port ttyUSB0 --baud 57600```: 

```--path_to_video rtsp://192.168.144.10:8554/H264Video```: Path the to stream.

```--fly_drone```: Indication to fly the drone.

```--port ttyUSB0```: The used port for connecting to the drone.

```--baud 57600```: Baud rate.

    
