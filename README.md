# Flask-Based Image Processing Application
This project is a Flask-based web application that uses YOLO and SAM models to perform object detection and segmentation on uploaded images. The application processes the uploaded image, detects relevant objects, crops each object, and generates segmentation masks. These images are then combined into a collage and presented to the user along with the processed images.

# Features
Image Upload: Users can upload image files through the web interface.
Object Detection and Segmentation: Detects objects in uploaded images and performs segmentation using the SAM model.
Visualization of Results: Combines detected objects with segmentation masks and presents them both individually and in a collage format.
Downloadable Files: Processed images can be downloaded by the user.

# Requirements
Python 3.x
Flask
OpenCV
NumPy
Ultralytics YOLO
Roboflow API
Supervision
Matplotlib
Open3D
Installation
Clone this repository to your local machine:

bash
git clone https://github.com/yourusername/flask-image-processing-app.git
cd flask-image-processing-app
Install the required Python packages:

bash

pip install -r requirements.txt
Create the necessary folders for UPLOAD_FOLDER and PROCESSED_FOLDER:

bash

mkdir upload_path processed_path
Download and place the required YOLO and SAM models in the directories specified by yolo_model_path and sam_model_path.

Set up the inference.py file and add your Roboflow API key.

# File Structure
app.py: The main file for the Flask application.
templates/: Contains the HTML files.
static/: Contains static files (CSS, JS).
upload_path/: Stores uploaded images.
processed_path/: Stores processed images.

# License
This project is licensed under the MIT License. 
https://lnkd.in/d2GQutwY
Author: Rithul
Type: { Open Source Dataset }

# You can access my Linkedin profile here:
https://www.linkedin.com/in/şafak-yaşlak-98a2b5217/
