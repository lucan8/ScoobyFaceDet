import os
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import cv2 as cv
import shutil
from collections import defaultdict

char_label_dict = {"daphne": 0, "fred": 1, "shaggy": 2, "velma": 3, "unknown": 4}
label_char_dict = {0: "daphne", 1: "fred", 2: "shaggy", 3: "velma", 4: "unknown"}
detection_size = 4

env_dir = os.path.dirname(os.path.dirname(__file__))
base_dir = os.path.dirname(os.path.dirname(env_dir))
train_dir = Path(os.path.join(base_dir, "antrenare"))
val_dir = Path(os.path.join(base_dir, "validare"))

def setup_yolo_label_files(global_data_dir, handle_ann_file_func):
    print(f"Using ann handler {handle_ann_file_func}")

    global_data_dir = Path(global_data_dir)
    for char_entity in global_data_dir.iterdir():
        if char_entity.is_file() and ("annotations" in char_entity.stem or "gt_validare" in  char_entity.stem):
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
        xc, yc, w, h = bbox_to_yolo(bboxes[i], img_h, img_w)
        label_line = f"{char_label_dict[char_names[i]]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        res += label_line + "\n"

    return res


def bbox_to_yolo(bbox, img_h, img_w):
    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]

    h, w = y1 - y0, x1 - x0
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2

    return xc / img_w, yc / img_h, w / img_w, h / img_h


def write_to_list_file(data_dir, list_file):
    global base_dir, env_dir
    data_dir = Path(data_dir)

    for file_path in data_dir.iterdir():
        if file_path.suffix == ".jpg":
            print(f"Handling file {file_path}")
            file_path_relative = file_path.relative_to(base_dir)
            list_file.write(f"{file_path_relative}\n")

def setup_train_list_file(train_dir):
    list_file_name = os.path.join(train_dir, "train.txt")
    if os.path.exists(list_file_name):
        os.remove(list_file_name)

    list_file = open(list_file_name, 'w')
    for entity in train_dir.iterdir():
        if entity.is_dir():
            print(f"Handling dir {entity}")
            write_to_list_file(entity, list_file)
    list_file.close()
    return list_file_name

def setup_val_list_file(val_dir):
    list_file_name = os.path.join(val_dir, "val.txt")
    if os.path.exists(list_file_name):
        os.remove(list_file_name)

    list_file = open(list_file_name, 'w')
    write_to_list_file(os.path.join(val_dir, "validare"), list_file)
    list_file.close()
    return list_file_name

def _handle_char_ann_file(data_dir, gt_detections, gt_file_names, gt_char_names):
    # If a string is given all detections are for a given character
    if isinstance(gt_char_names, str):
        gt_char_names = np.full(len(gt_detections), gt_char_names)

    last_ind = 0
    for i in range(1, len(gt_file_names)):
        if gt_file_names[i] != gt_file_names[last_ind]:
            # Read image for shape
            img_file_path = os.path.join(data_dir, gt_file_names[last_ind])
            # Ensure the image is read correctly and has a shape attribute
            img = cv.imread(img_file_path)
            if img is None:
                print(f"Warning: Could not read image {img_file_path}. Skipping.")
                last_ind = i
                continue
            img_h, img_w, _ = img.shape

            # Prepare label file
            label_file_path = Path(gt_file_names[last_ind]).stem + ".txt"
            label_file_path = os.path.join(data_dir, label_file_path)
            print(f"Handling image file {label_file_path}")

            # Get label string for this image file
            detections, char_names = gt_detections[last_ind:i], gt_char_names[last_ind:i]
            res_str = det_to_str(detections, char_names, img_w, img_h)

            # Write result to file
            with open(label_file_path, "w") as f:
                f.write(res_str)

            last_ind = i

    # Handle the last set of detections
    if len(gt_file_names) > 0:
        img_file_path = os.path.join(data_dir, gt_file_names[last_ind])
        img = cv.imread(img_file_path)
        if img is None:
            print(f"Warning: Could not read image {img_file_path}. Skipping final detection.")
        else:
            img_h, img_w, _ = img.shape

            label_file_path = Path(gt_file_names[last_ind]).stem + ".txt"
            label_file_path = os.path.join(data_dir, label_file_path)
            print(f"Handling image file {label_file_path}")

            detections, char_names = gt_detections[last_ind:], gt_char_names[last_ind:]
            res_str = det_to_str(detections, char_names, img_w, img_h)

            with open(label_file_path, "w") as f:
                f.write(res_str)

def save_all_detections(detections: np.ndarray, scores: np.ndarray, file_names: np.ndarray, save_dir: str):
        np.save(os.path.join(save_dir, "detections_all_faces"), detections)
        np.save(os.path.join(save_dir, "scores_all_faces"), scores)
        np.save(os.path.join(save_dir, "file_names_all_faces"), file_names)

def save_detections_char(detections: np.ndarray, scores: np.ndarray, file_names: np.ndarray, char_save_dir: str):
    if not os.path.exists(char_save_dir):
        os.makedirs(char_save_dir)

    np.save(os.path.join(char_save_dir, "detections.npy"), detections)
    np.save(os.path.join(char_save_dir, "scores.npy"), scores)
    np.save(os.path.join(char_save_dir, "file_names.npy"), file_names)

def save_detections_chars(detections: dict[str, list[np.ndarray]], save_dir: str):
    for char_name in detections:
        char_test_res_data_dir = os.path.join(save_dir, char_name)
        save_detections_char(detections[char_name][0], detections[char_name][1], detections[char_name][2], char_test_res_data_dir)

# Returns 4 lists, bboxes, scores, filenames and labels for all yolo's detections
def yolo_detections_to_my_format(results):
  all_bboxes = []
  all_filenames = []
  all_scores = []
  all_class_labels = []

  for result in results:
      filename = os.path.basename(result.path)
      if result.boxes:
          for box in result.boxes:
              all_bboxes.append(box.xyxy[0].cpu().numpy())
              all_scores.append(box.conf.cpu().numpy()[0])
              all_filenames.append(filename)
              all_class_labels.append(box.cls.cpu().numpy().item())

  return np.array(all_bboxes), np.array(all_scores), np.array(all_filenames), np.array(all_class_labels)

def split_detections_by_char(bboxes: np.ndarray, scores: np.ndarray, file_names: np.ndarray, class_labels: np.ndarray):
  detections_by_char = defaultdict(lambda: [[], [], []]) # Stores lists of [bboxes, scores, filenames]

  for i in range(len(bboxes)):
      class_id = class_labels[i]
      char_name = label_char_dict[class_id]

      detections_by_char[char_name][0].append(bboxes[i])
      detections_by_char[char_name][1].append(scores[i])
      detections_by_char[char_name][2].append(file_names[i])

  # Convert lists within detections_by_char to NumPy arrays for consistency
  for char_name in detections_by_char:
      detections_by_char[char_name][0] = np.array(detections_by_char[char_name][0])
      detections_by_char[char_name][1] = np.array(detections_by_char[char_name][1])
      detections_by_char[char_name][2] = np.array(detections_by_char[char_name][2])

  return detections_by_char

def run_project():
    global train_dir, val_dir
    setup_yolo_label_files(train_dir, handle_train_char_ann_file)
    setup_yolo_label_files(val_dir, handle_val_char_ann_file)

    setup_train_list_file(train_dir)
    setup_val_list_file(val_dir)

    val_dir = val_dir / "validare"

    bonus_path = Path(os.path.join(env_dir, "saved_files", "bonus"))
    os.makedirs(bonus_path, exist_ok=True)

    # Load model (retrain if needed)
    model_dir = Path(os.path.join(bonus_path, "models"))
    os.makedirs(model_dir, exist_ok=True)
    model_file = str(model_dir / "best.pt")

    if not os.path.exists(model_file):
        print(f"Model file {model_file} not found, retraining...")
        YOLO("yolov8n.pt").train(
            data=os.path.join(base_dir, "data.yaml"),
            epochs=10,
            imgsz=640,
            batch=16,
            lr0=0.01,
            device="cpu" # Change to 0 if gpu available
        )
        print(f"Saving to {model_file}")
        shutil.move(os.path.join(base_dir, "runs", "detect", "train", "weights", "best.pt"), model_file)

    model = YOLO(model_file)

    # Run predictions and collect results
    print("Running model predictions on the validation set...")
    results_predict = model.predict(source=val_dir, save=False)

    all_bboxes, all_scores, all_filenames, all_labels = yolo_detections_to_my_format(results_predict)

    print(all_bboxes.shape)
    print(all_scores.shape)
    print(all_filenames.shape)
    print(all_labels.shape)

    detections_by_char = split_detections_by_char(all_bboxes, all_scores, all_filenames, all_labels)

    # Define the new output directory for all results
    output_base_dir = os.path.join(bonus_path, 'results', 'data')
    os.makedirs(output_base_dir, exist_ok=True)

    print(f"Saving global detection results to: {output_base_dir}")
    save_all_detections(all_bboxes, all_scores, all_filenames, output_base_dir)

    print(f"Saving character-specific detection results within subdirectories of: {output_base_dir}")
    save_detections_chars(detections_by_char, output_base_dir)

run_project()