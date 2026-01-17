import os
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import cv2 as cv
char_label_dict = {"daphne": 0, "fred": 1, "shaggy": 2, "velma": 3, "unknown": 4}
label_char_dict = {0: "daphne", 1: "fred", 2: "shaggy", 3: "velma", 4: "unknown"}
detection_size = 4

                
def setup_yolo_label_files(global_data_dir, handle_ann_file_func):
    print(f"Using ann handler {handle_ann_file_func}")

    global_data_dir = Path(global_data_dir)
    for char_entity in global_data_dir.iterdir():
        if char_entity.is_file():
            print(f"Handling annotation file {char_entity}")
            handle_ann_file_func(char_entity)

def handle_val_char_ann_file(char_ann_file):
    global detection_size
    ann_file_name = char_ann_file.stem
    ann_file_split = ann_file_name.split("_")
    task, char_name = ann_file_split[0], ann_file_split[1]

    # Only care about per char annotation files
    if task == "task1":
        return

    ground_truth_file = np.loadtxt(char_ann_file, dtype='str')
    gt_file_names = np.array(ground_truth_file[:, 0])
    gt_detections = np.array(ground_truth_file[:, 1:detection_size + 1], int)

    val_img_dir = os.path.join(os.path.dirname(char_ann_file), "validare")
    _handle_char_ann_file(val_img_dir, gt_detections, gt_file_names, char_name)
    

def handle_train_char_ann_file(char_ann_file):
    global detection_size

    ground_truth_file = np.loadtxt(char_ann_file, dtype='str')
    gt_file_names = np.array(ground_truth_file[:, 0])
    gt_detections = np.array(ground_truth_file[:, 1:detection_size + 1], int)
    gt_char_names = np.array(ground_truth_file[:, detection_size + 1])

    char_name = char_ann_file.stem.split("_")[0]
    char_dir = os.path.join(os.path.dirname(char_ann_file), char_name)

    _handle_char_ann_file(char_dir, gt_detections, gt_file_names, gt_char_names)

def det_to_str(bboxes, char_names, img_w, img_h):
    global label_char_dict, char_label_dict
    res = ""
    for i in range(len(bboxes)):
        xc, yc, w, h = bbox_to_yolo(bboxes[i], img_w, img_h)
        label_line = f"{char_label_dict[char_names[i]]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        res += label_line + "\n"
    
    return res

def bbox_to_yolo(bbox, img_h, img_w):
    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]

    h, w = y1 - y0, x1 - x0
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2

    return xc / img_w, yc / img_h, w / img_w, h / img_h


def write_to_list_file(data_dir, list_file):
    global base_dir
    data_dir = Path(data_dir)

    for file_path in data_dir.iterdir():
        if file_path.suffix == ".jpg":
            print(f"Handling file {file_path}")
            file_path = file_path.relative_to(base_dir)
            list_file.write(f"{file_path}\n")

def setup_train_list_file(train_dir):
    list_file_name = os.path.join(train_dir, "train.txt")
    if os.path.exists(list_file_name):
        print("Train file list file already setup!")
        return
    
    list_file = open(list_file_name, 'w')
    for entity in train_dir.iterdir():
        if entity.is_dir():
            print(f"Handling dir {entity}")
            write_to_list_file(entity, list_file)
    
    return list_file

def setup_val_list_file(val_dir):
    list_file_name = os.path.join(val_dir, "val.txt")
    if os.path.exists(list_file_name):
        print("Train file list file already setup!")
        return
    
    list_file = open(list_file_name, 'w')
    write_to_list_file(os.path.join(val_dir, "validare"), list_file)
    
    return list_file

def _handle_char_ann_file(data_dir, gt_detections, gt_file_names, gt_char_names):
    # If a string is given all detections are for a given character
    if isinstance(gt_char_names, str):
        gt_char_names = np.full(len(gt_detections), gt_char_names)

    last_ind = 0
    for i in range(1, len(gt_file_names)):
        if gt_file_names[i] != gt_file_names[last_ind]:
            # Read image for shape
            img_file_path = os.path.join(data_dir, gt_file_names[last_ind])
            img_h, img_w, _ = cv.imread(img_file_path, cv.IMREAD_GRAYSCALE).shape
           
            # Prepare label file
            label_file_path = Path(gt_file_names[last_ind]).stem + ".txt"
            label_file_path = os.path.join(data_dir, label_file_path)
            print(f"Handling image file {label_file_path}")
            
            # Get label string for this image file
            detections, char_names = gt_detections[last_ind:i], gt_char_names[last_ind:i]
            res_str = det_to_str(detections, char_names, img_h, img_w)

            # Write result to file
            open(label_file_path, "a").write(res_str)

            last_ind = i

    # Read image for shape
    img_file_path = os.path.join(data_dir, gt_file_names[last_ind])
    img_h, img_w, _ = cv.imread(img_file_path, cv.IMREAD_GRAYSCALE).shape

    # Prepare label file
    label_file_path = Path(gt_file_names[last_ind]).stem + ".txt"
    label_file_path = os.path.join(data_dir, label_file_path)
    print(f"Handling image file {label_file_path}")
    
    # Get label string for this image file
    detections, char_names = gt_detections[last_ind:], gt_char_names[last_ind:]
    res_str = det_to_str(detections, char_names, img_h, img_w)

    # Write result to file
    open(label_file_path, "w").write(res_str)

curr_dir = os.path.dirname(__file__)
base_dir = os.path.dirname(curr_dir)
train_dir = Path(os.path.join(base_dir, "antrenare"))
val_dir = Path(os.path.join(base_dir, "validare"))

# setup_yolo_label_files(val_dir, handle_val_char_ann_file)

# setup_train_file_list_file(train_dir)
# setup_val_list_file(val_dir)

# Nano model (fast, good starting point)
model = YOLO("yolov8n.pt")

model.train(
    data=os.path.join(base_dir, "data.yaml"),
    epochs=1,
    imgsz=640,
    batch=16,
    lr0=0.01,
    device='cpu'   # GPU 0; use "cpu" if no GPU
)