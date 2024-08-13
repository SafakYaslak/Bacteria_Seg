from flask import Flask, request, render_template, send_from_directory
import os
import cv2
import numpy as np
from ultralytics import YOLO, SAM
from roboflow import Roboflow
import supervision as sv

from inference import get_model
app = Flask(__name__)


UPLOAD_FOLDER = r'upload_path'
PROCESSED_FOLDER = r'processed_path'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load the YOLO and SAM models
yolo_model_path = r"yolo_model_path" 
sam_model_path = r"sam_model_path"  
# model = YOLO(yolo_model_path)
sam_model = SAM(sam_model_path)


model = get_model(model_id="bacteria-detection-qhihf/3", api_key="api_key")
def create_collage(images, images_per_row):
    if len(images) == 0:
        return None

    height, width = images[0].shape[:2]
    num_images = len(images)
    num_rows = (num_images + images_per_row - 1) // images_per_row
    collage = np.zeros((num_rows * height, images_per_row * width, 3), dtype=np.uint8)
    
    for i, img in enumerate(images):
        row = i // images_per_row
        col = i % images_per_row
        collage[row * height:(row + 1) * height, col * width:(col + 1) * width] = img
    
    return collage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the image
        image_pairs, collage_path, image_with_boxes_path= process_image(file_path)
        
        return render_template('result.html', original=filename, image_pairs=image_pairs, collage=collage_path, image_with_boxes=image_with_boxes_path, enumerate=enumerate)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

def process_image(image_path):
    image = cv2.imread(image_path)
    kolaj_w, kolaj_h = 200, 150
    new_width, new_height = 800, 600
    results = model.infer(image)[0]
  
    min_area_threshold = 20
    combined_images = []
    image_pairs = []  
    image_with_boxes = image.copy()

    boxes =[]
    for  result in results.predictions:
        x = int(result.x)
        y = int(result.y)
        width = int(result.width)
        height = int(result.height)

        
        # Create a new row
        boxes.append([x, y, width, height])
        print(result)
        print(boxes)

    for i, box in enumerate(boxes):

        
        start_point = (int(box[0] - box[2] / 2), int(box[1] - box[3] / 2))
        end_point = (int(box[0] + box[2]/ 2), int(box[1] + box[3] / 2))

        cropped_image = image[int(box[1] - box[3] / 2):int(box[1] + box[3] / 2), int(box[0] - box[2] / 2):int(box[0] + box[2]/ 2)]
        resized_image = cv2.resize(cropped_image, (new_width, new_height))

        results_sam = sam_model(resized_image, points=[400,300], labels=[1], show=False)
        # results_sam = sam_model(resized_image, bboxes=[0,0,new_width,new_height], labels=[1], show=False)
        masks = results_sam[0].masks
        image_with_boxes = cv2.rectangle(image_with_boxes, start_point , end_point,  (0, 0, 255), 4)

        if masks is not None:
            masks_data = masks.data.cpu().numpy()

            mask_overlay = np.zeros_like(resized_image)
            mask_canvas = np.zeros_like(resized_image)
            
            for mask in masks_data:
                binary_mask = (mask > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    contour_area = cv2.contourArea(contour)
                    if contour_area > min_area_threshold:
                        mask_canvas = np.zeros_like(resized_image)
                        cv2.drawContours(mask_canvas, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

                        
                        object_extracted = cv2.bitwise_and(resized_image, mask_canvas)

                       
                        black_canvas = np.zeros_like(resized_image)
                        black_canvas[mask_canvas > 0] = object_extracted[mask_canvas > 0]

                        cv2.drawContours(mask_overlay, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)
                        cv2.drawContours(mask_overlay, [contour], -1, (0, 0, 255), thickness=7)

                        cv2.drawContours(mask_canvas, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)
                        cv2.drawContours(mask_canvas, [contour], -1, (0, 0, 255), thickness=7)

            combined_image = cv2.addWeighted(resized_image, 1.0, mask_overlay, 1.0, 0)
            
            cropped_image_copy = cv2.resize(cropped_image.copy(), (kolaj_w, kolaj_h))
            combined_image_copy = cv2.resize(combined_image.copy(), (kolaj_w, kolaj_h))
            black_canvas_copy = cv2.resize(black_canvas.copy(), (kolaj_w, kolaj_h))
            mask_canvas_copy = cv2.resize(mask_canvas.copy(), (kolaj_w, kolaj_h))

            birlesik = cv2.hconcat([cropped_image_copy, combined_image_copy])
            combined_images.append(birlesik)

            # Save the cropped and combined images
            cropped_image_path = os.path.join(app.config['PROCESSED_FOLDER'], f'Detected_Object_{i}.png')
            combined_image_path = os.path.join(app.config['PROCESSED_FOLDER'], f'Segmented_Image_{i}.png')
            mask_path = os.path.join(app.config['PROCESSED_FOLDER'], f'Masked_Image_{i}.png')
            black_canvas_path = os.path.join(app.config['PROCESSED_FOLDER'], f'Black_Canvas_{i}.png')

            cv2.imwrite(cropped_image_path, cropped_image_copy)
            cv2.imwrite(combined_image_path, combined_image_copy)
            cv2.imwrite(mask_path, mask_canvas_copy)
            cv2.imwrite(black_canvas_path, black_canvas_copy)
        
            image_pairs.append((f'Detected_Object_{i}.png', f'Segmented_Image_{i}.png', f'Masked_Image_{i}.png', f'Black_Canvas_{i}.png'))

    collage = create_collage(combined_images, 4)
    collage = cv2.resize(collage, (image.shape[1], image.shape[0]))
    collage_path = None
    if collage is not None:
        collage_path = 'collage.png'
        collage_full_path = os.path.join(app.config['PROCESSED_FOLDER'], collage_path)
        cv2.imwrite(collage_full_path, collage)

    image_with_boxes_path = None
    if image_with_boxes is not None:
        image_with_boxes_path = 'image_with_boxes.png'
        image_with_boxes_full_path = os.path.join(app.config['PROCESSED_FOLDER'], image_with_boxes_path)
        cv2.imwrite(image_with_boxes_full_path, image_with_boxes)

    return image_pairs, collage_path, image_with_boxes_path


if __name__ == '__main__':
    app.run(debug=True)