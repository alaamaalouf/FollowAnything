import cv2
import os
import warnings
import numpy as np
import torch
import torchvision
torch.hub.set_dir("./hub")

def preprocess_image(img, half = False,reshape_to = None):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if reshape_to: 
        img = cv2.resize(img, reshape_to)
    img = torch.from_numpy(img).float() / 255.0
    img = img[..., :3]  # drop alpha channel, if present
    img = img.cuda()
    img = img.permute(2, 0, 1)  # C, H, W
    img = img.unsqueeze(0)  # 1, C, H, W
    img = img.cuda()
    if half:
        img = img.half()

    return img


def upsample_feat_vec(feat, target_shape):
    return torch.nn.functional.interpolate(
        feat, target_shape, mode="bilinear", align_corners=True
    )


class VITFeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        model_type="dino_vits8",
        stride=4,
        device="cuda:0",
        load_size=224,
        upsample=False,
        **kwargs,
    ):
        from .dino_feature_extractor import ViTExtractor

        super().__init__()
        self.extractor = ViTExtractor(model_type, stride, device=device)
        self.load_size = load_size
        self.input_image_transform = self.get_input_image_transform()
        if upsample == True:
            if "desired_height" in kwargs.keys():
                self.desired_height = kwargs["desired_height"]
                if "desired_width" in kwargs.keys():
                    self.desired_width = kwargs["desired_width"]
                    self.upsample = True
                else:
                    warnings.warn(
                        "Ignoring upsample arguments as they are incomplete. "
                        "Missing `desired_width`."
                    )
            else:
                warnings.warn(
                    "Ignoring upsample arguments as they are incomplete. "
                    "Missing `desired_height`."
                )
        else:
            self.upsample = False
        # Layer to extract feature maps from
        self.layer_idx_to_extract_from = 11
        if "layer" in kwargs.keys():
            self.layer_idx_to_extract_from = kwargs["layer"]
        # Type of attention component to create descriptors from
        self.facet = "key"
        if "facet" in kwargs.keys():
            self.facet = kwargs["facet"]
        # Whether or not to create a binned descriptor
        self.binned = False
        if "binned" in kwargs.keys():
            self.binned = kwargs["binned"]

    def get_input_image_transform(self):
        _NORM_MEAN = [0.485, 0.456, 0.406]
        _NORM_STD = [0.229, 0.224, 0.225]
        return torchvision.transforms.Compose(
            [torchvision.transforms.Normalize(mean=_NORM_MEAN, std=_NORM_STD)]
        )

    def forward(self, img, apply_default_input_transform=False):
        img = torchvision.transforms.functional.resize(img, self.load_size)
        #
        #print("shape is", img.shape)
        #img = torchvision.transforms.functional.resize(img, (self.load_size, int(img.shape[-1]/(img.shape[-2]/self.load_size))))
         
        #print(img.shape)
        #input()
        if apply_default_input_transform:
            # Default input image transfoms
            img = self.input_image_transform(img)
        feat = self.extractor.extract_descriptors(
            img, self.layer_idx_to_extract_from, self.facet, self.binned
        )
        feat = feat.reshape(
            self.extractor.num_patches[0],
            self.extractor.num_patches[1],
            feat.shape[-1],
        )
        feat = feat.permute(2, 0, 1)
        feat = feat.unsqueeze(0)
        if self.upsample:
            feat = upsample_feat_vec(feat, [self.desired_height, self.desired_width])
        return feat

def binary_boundaries(labels, cutoff=0.5):  
  return [consecutive(channel.nonzero()[0]) for channel in binarize(labels, cutoff)]


