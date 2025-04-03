import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

cfg = get_cfg()

cfg.merge_from_file("path/to/your/config.yaml") 

cfg.MODEL.WEIGHTS =  "path/to/your/model_final.pth"  
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST =  0.5  



### creating predector
predictor = DefaultPredictor(cfg)

def segment_image(image):
    """
    Runs inference on the input image and returns a segmented output.
    """
    outputs = predictor(image)
    visualizer = Visualizer(image[:, :, ::-1], scale=1.2)
    vis_output = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_img = vis_output.get_image()[:, :, ::-1]
    return result_img
