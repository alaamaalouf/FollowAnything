# Follow Anything

**[1 CSAIL, MIT](https://www.csail.mit.edu/)**  | **[2 SEAS, Harvard](https://www.csail.mit.edu/](https://seas.harvard.edu/)https://seas.harvard.edu/)**

*[Alaa Maalouf](https://www.csail.mit.edu/person/alaa-maalouf), [Ninad Jadhav](https://react.seas.harvard.edu/people/ninad-jadhav), [Krishna Murthy Jatavallabhula](https://krrish94.github.io/), [Makram Chahine](https://www.mit.edu/~chahine/), [Daniel  M.Vogt](https://www.danielmvogt.com/), [Robert J. Wood](https://wyss.harvard.edu/team/associate-faculty/robert-wood/), [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/), and [Daniela Rus](https://danielarus.csail.mit.edu/)*

![SAM design](Images_and_videos_for_Github_visualizations/teaser.png?raw=true)



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

        a. SAM model to Segment-and-Track-Anything/ckpt directory, the default model is SAM-VIT-B (sam_vit_b_01ec64.pth).
    
        b. DeAOT/AOT model to Segment-and-Track-Anything/ckpt directory, the default model is R50-DeAOT-L (R50_DeAOTL_PRE_YTB_DAV.pth).


    Note - some files are slightly modified in the directory Segment-and-Track-Anything, hence, use the version provided in this directory.
    

 4. If you wish to use SiamMask as a tracker (default is AOT from step "2") install SiamMask as detailed in: https://github.com/foolwood/SiamMask

     Note - some files are slightly modified in the directory Segment-and-Track-Anything, hence, use the version provided in this directory.

 5. pip install mavsdk (you may need to do more simple pip installs for other libraries).

    
