from DINO.collect_dino_features import *
import cv2

def preprocess_frame(img, cfg):
    #if cfg['desired_height'] > 480 or cfg['desired_width']>640:
    #print("cfg['desired_width'],cfg['desired_height']", cfg['desired_width'],cfg['desired_height'] )
    return preprocess_image(img, half = cfg['use_16bit'],reshape_to = (cfg['desired_width'], cfg['desired_height'] ))

    #return preprocess_image(img, half = cfg['use_16bit'],reshape_to = (640, 480))
   
def get_dino_pixel_wise_features_model(cfg, device):

    ## See DINO.collect_dino_features
    model = VITFeatureExtractor(
        upsample= True,
        stride=cfg['dino_strides'], #default 4
        desired_height=cfg['desired_height'], # default 240,
        desired_width=cfg['desired_width'], #320,
       
    )

    model.extractor.model.eval().to(device)
    example_input = torch.randn(1, 3, cfg['desired_height'], cfg['desired_width']).to(device) #Todo: should be removed?

    if cfg['use_16bit']:
        model.extractor.model.half()
        model.half()
        example_input = example_input.half()

    model.extractor.model.eval()
    model.eval()
    model.to(device)

    for name, para in model.extractor.model.named_parameters():
            para.requires_grad = False
    
    if cfg['use_traced_model']:
        model = torch.jit.trace(model, example_input)
    
    return model
