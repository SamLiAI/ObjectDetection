import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Load the model
model_dir = 'ssd_mobilenet_v2_320x320_coco17_tpu-8/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model'
model = tf.saved_model.load(str(model_dir))
model = model.signatures['serving_default']

# Load label map
PATH_TO_LABELS = 'models-master/research/object_detection/data/mscoco_label_map.pbtxt'  # Update this path
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image_path):
    image = Image.open(image_path)
    return np.array(image)

def run_inference_for_single_image(model, image_np):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    return detections

def detect_human(image_path):
    image_np = load_image_into_numpy_array(image_path)
    detections = run_inference_for_single_image(model, image_np)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.5,
        line_thickness=2)
    return image_np

# Test the function


image_path = 'photos/people.jpg'  # Update this path
output_np = detect_human(image_path)
output_image = Image.fromarray(output_np)
output_image.show()