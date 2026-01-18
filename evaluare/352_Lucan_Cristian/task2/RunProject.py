from Parameters import *
from FacialClassifier import *
import pdb
from pathlib import Path
import shutil
import os
from math import log, isclose
import hashlib

# BEST MODEL: d0ff06c1b25c5189492639f8734faefe
# SECOND MDOEL: 6dc9dc55739c7c6471fc095f4f1fd43a
# THIRD BEST MODEL: a8fb0d0787a6e359e8fceae8b3d3b6e0
# FOURTH: 144ad3072484aa023e76c7410f99c65a

def print_dict_size(dic):
    for k, v in dic.items():
        print(f"{k}: {len(v)}")
    print()

def train_detector(detector: FacialClassifer):
    # Make the detector horizon bigger
    detector.params.dim_window_upper += detector.params.tolerance_upper
    detector.params.dim_window_lower -= detector.params.tolerance_lower
    detector.params.set_run_dirs()

    # Get the positive and negative descriptors
    train_desc = detector.get_train_desc()
    print(f"Fetched descriptors!")
    print_dict_size(train_desc)
    
    # Restore horizon
    detector.params.dim_window_upper -= detector.params.tolerance_upper
    detector.params.dim_window_lower += detector.params.tolerance_lower
    detector.params.set_run_dirs()
    
    # Train classifier
    training_examples, train_labels = detector.get_merged_data(train_desc)
    print(f"Merged training examples and labels!")

    detector.train_classifier(training_examples, train_labels)

def run_detector(detector: FacialClassifer, all_detections, all_files):
    # Test model
    print("Running detector...")
    start_time = timeit.default_timer()
    detections = detector.run(all_detections, all_files)
    end_time = timeit.default_timer()
    print(f"Running detector took: {end_time - start_time} sec!")

    return detections

def run_project():
    facial_detector: FacialClassifer = FacialClassifer(Parameters())

    facial_detector.params.threshold = 0
    facial_detector.params.use_cache = False
    facial_detector.params.hard_pos_overlap_step = 0.00001
  
    # Window parameters, medium window
    # The "neg" parameters don't matter here, I was too lazy to remove them from the earlier model
    facial_detector.best_model = None
    facial_detector.params.hard_neg_mining_it_count = 4
    facial_detector.params.neg_patch_factor = 2.5
    facial_detector.params.soft_neg_overlap = 0.25
    facial_detector.params.dim_hog_cell = 6
    facial_detector.params.set_window_stuff(64, 170, 0.9, 1.3, 18)
    facial_detector.params.set_run_dirs()

    train_detector(facial_detector)

    # Fetch detections of best face/no-face model
    # IF YOU DECIDE TO RE-TRAIN THE FACE DETECTORS YOU WILL HAVE TO CHANGE THE HASH!
    best_model_det_dir = os.path.join(facial_detector.params.dir_save_files, "merge_results", "3fa93ef7c172591e58699e9ad51a72ec", "data")

    all_det, all_scores, all_files = facial_detector.fetch_all_detections(best_model_det_dir)
    print(f"Fetched all {len(all_det)} detections!")

    # Run the classifier on the detections
    detections = facial_detector.run(all_det, all_files)

    # Set up files and dirs
    merged_fd_name = "_".join(best_model_det_dir.split('\\')[-3:-1]) + "\n" + facial_detector.get_name()
    merged_fd_name_hash = hashlib.md5(merged_fd_name.encode()).hexdigest()
    merge_dir = os.path.join(facial_detector.params.dir_save_files, "merge_results_classifier", merged_fd_name_hash)

    # to more easily identify the model(s) used
    merge_info_file = os.path.join(merge_dir, "info.txt")
    os.makedirs(merge_dir, exist_ok=True)
    open(merge_info_file, 'w').write(merged_fd_name)

    # Save detections
    facial_detector.save_detections(detections, merge_dir)

    # Write additional information and evaluate detections
    # facial_detector.eval_detections_split(detections, merge_dir)

run_project()