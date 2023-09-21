# Object Detection using TensorFlow and OpenCV

## Description

This project aims to perform object detection using TensorFlow's SSD MobileNet and OpenCV. It is specifically designed to detect humans in images but can be easily extended to detect other objects.

### Table of Contents
#### Installation
#### Usage
#### License


### Installation
#### Clone the repository:

        git clone https://github.com/your-username/your-repo-name.git

#### Install the required packages:
        pip install -r requirements.txt
        
 #### Download 
 the pre-trained model and place it in the ssd_mobilenet_v2_320x320_coco17_tpu-8/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model directory.

 #### Update 
 the PATH_TO_LABELS and image_path in the code to point to the correct locations.

### Usage
Run the main script to perform object detection on an image:

        python your_script_name.py
This will display the image with bounding boxes around detected humans.


### License
This project is licensed under the MIT License - see the LICENSE.md file for details.
