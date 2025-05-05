import os
import uuid
import json
import subprocess
import zipfile
import io
from flask import Flask, request, render_template, send_file, redirect, url_for
from forensics_check import extract_exif, compute_hashes, perform_ela
from datetime import datetime
import numpy as np
import cv2
import pytesseract as pt

# YOLO Configuration
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
net = cv2.dnn.readNetFromONNX('best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Flask app initialization
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
WEIGHTS_PATH = 'intw32_quantized_model_10000.pth'
PYTHON_PATH = 'ImageEnhancer/bin/python'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# YOLO functions
def get_detections(img, net):
    image = img.copy()
    row, col, d = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    return input_image, detections

def non_maximum_suppression(input_image, detections):
    boxes = []
    confidences = []
    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]  # confidence of detecting license plate
        if confidence > 0.4:
            class_score = row[5]  # probability score of license plate
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])

                confidences.append(confidence)
                boxes.append(box)

    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    index = np.array(cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)).flatten()
    return boxes_np, confidences_np, index

def extract_text(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y + h, x:x + w]
    if 0 in roi.shape:
        return ''
    else:
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        magic_color = apply_brightness_contrast(gray, brightness=40, contrast=70)
        text = pt.image_to_string(magic_color, lang='eng', config='--psm 6')
        text = text.strip()
        return text

def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def yolo_predictions(img, net):
    input_image, detections = get_detections(img, net)
    boxes_np, confidences_np, index = non_maximum_suppression(input_image, detections)
    result_img = img.copy()
    text_list = []
    for ind in index:
        x, y, w, h = boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf * 100)
        license_text = extract_text(img, boxes_np[ind])

        cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.rectangle(result_img, (x, y - 30), (x + w, y), (255, 0, 255), -1)
        cv2.rectangle(result_img, (x, y + h), (x + w, y + h + 30), (0, 0, 0), -1)
        cv2.putText(result_img, conf_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(result_img, license_text, (x, y + h + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

        text_list.append(license_text)

    # Save the OCR result image
    ocr_image_path = os.path.join(RESULT_FOLDER, 'ocr_' + str(uuid.uuid4().hex) + '.jpg')
    cv2.imwrite(ocr_image_path, result_img)

    return result_img, text_list, ocr_image_path  # Return the image path for OCR output

@app.route('/', methods=['GET', 'POST'])
def index():
    error_msg = None
    result_img = None
    ocr_img = None  # Add a variable for OCR output image
    vehicle_info = None
    exif_data = None
    hash_data = None
    ela_image = None
    image_processed = False
    image_filename = None
    forensics_checked = False

    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            if image.filename == '':
                error_msg = "No selected file"
            else:
                filename = f"{uuid.uuid4().hex}.jpg"
                image_path = os.path.join(UPLOAD_FOLDER, filename)
                result_path = os.path.join(RESULT_FOLDER, f"deblur_{filename}")
                image.save(image_path)

                try:
                    # Run the deblurring script
                    subprocess.run(
                        [sys.executable, 'test.py',
                         '--input_img', image_path,
                         '--output_img', result_path,
                         '--weights', WEIGHTS_PATH],
                        check=True
                    )
                    result_img = result_path  # This is now the deblurred image
                    image_processed = True
                    image_filename = filename

                    # Now process the deblurred image with YOLO
                    img = cv2.imread(result_img)  # Use deblurred image
                    result_img, text_list, ocr_img_path = yolo_predictions(img, net)
                    ocr_img = ocr_img_path  # Set the OCR image path

                except subprocess.CalledProcessError as e:
                    error_msg = f"Deblurring failed: {e}"

        elif 'car_number' in request.form:
            car_number = request.form['car_number']
            image_filename = request.form['image_filename']
            result_img = os.path.join(RESULT_FOLDER, f"deblur_{image_filename}")
            try:
                import http.client
                conn = http.client.HTTPSConnection("vehicle-information-verification-rto-india.p.rapidapi.com")
                payload = json.dumps({"id_number": car_number})
                headers = {
                    'x-rapidapi-key': "3ad00cd098msh3b8991e10468228p135f12jsn55bd4e32d378",
                    'x-rapidapi-host': "vehicle-information-verification-rto-india.p.rapidapi.com",
                    'Content-Type': "application/json"
                }
                conn.request("POST", "/rc-full", payload, headers)
                res = conn.getresponse()
                data = res.read()
                vehicle_info = json.loads(data.decode("utf-8"))
                image_processed = True
            except Exception as e:
                error_msg = f"Vehicle API error: {str(e)}"

        elif 'forensics_check' in request.form:
            image_filename = request.form['image_filename']
            image_path = os.path.join(UPLOAD_FOLDER, image_filename)

            # Perform EXIF extraction, hash computation, and ELA check
            exif_data = extract_exif(image_path)
            hash_data = compute_hashes(image_path)
            ela_image = perform_ela(image_path)

            result_img = os.path.join(RESULT_FOLDER, f"deblur_{image_filename}")
            image_processed = True
            forensics_checked = True

    return render_template('index.html',
                           error_msg=error_msg,
                           result_img=result_img,  # Now this is a file path string
                           ocr_img=ocr_img,  # Pass OCR image path
                           vehicle_info=vehicle_info,
                           exif_data=exif_data,
                           hash_data=hash_data,
                           ela_image=ela_image,  # Pass ELA image path
                           image_processed=image_processed,
                           image_filename=image_filename,
                           forensics_checked=forensics_checked)

@app.route('/results')
def results():
    result_files = []
    # Get all files in the 'RESULT_FOLDER' and sort them by timestamp
    for filename in os.listdir(RESULT_FOLDER):
        file_path = os.path.join(RESULT_FOLDER, filename)
        if os.path.isfile(file_path):
            timestamp = os.path.getmtime(file_path)
            timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            result_files.append({'filename': filename, 'timestamp': timestamp_str})

    # Sort by timestamp
    result_files.sort(key=lambda x: x['timestamp'], reverse=True)

    return render_template('results.html', result_files=result_files)

@app.route('/delete_file/<filename>', methods=['POST'])
def delete_file(filename):
    file_path = os.path.join(RESULT_FOLDER, filename)

    try:
        # Check if the file exists
        if os.path.exists(file_path):
            os.remove(file_path)
            message = f"File {filename} deleted successfully."
        else:
            message = f"File {filename} not found."
        
        # Redirect back to the results page after deletion
        return redirect(url_for('results', message=message))
    
    except Exception as e:
        message = f"Error deleting file: {str(e)}"
        return redirect(url_for('results', message=message))

@app.route('/delete_all_files', methods=['POST'])
def delete_all_files():
    try:
        # Delete all files in the result folder
        for filename in os.listdir(RESULT_FOLDER):
            file_path = os.path.join(RESULT_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Redirect back to the results page after deletion
        return redirect(url_for('results'))
    
    except Exception as e:
        message = f"Error deleting files: {str(e)}"
        return redirect(url_for('results', message=message))
    
@app.route('/view_image/<filename>', methods=['GET'])
def view_image(filename):
    # Assuming you store images in a folder called 'static/images'
    try:
        # Path to the image you want to view
        image_path = os.path.join('static', 'results', filename)

        # Check if the file exists
        if not os.path.exists(image_path):
            return "Image not found.", 404

        # Return the image to be viewed directly in the browser
        return send_file(image_path, mimetype='image/jpeg')
    
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)
