from Parameters import *
from FacialDetector import *
import pdb
from pathlib import Path
import shutil
import os
from math import log, isclose
import hashlib

# TODO: IF THE IMAGE IS TOO SMALL FOR YOUR HOG YOU GET ERROR HANDLE IT CORRECTLY
# TODO: Try only one model for classification or at least expand the area of the existing classifiers!
# NAME IS USELESS WHEN MERGING PER CHARACTER

# BEST MODEL: f4197f5b96720b40d58d806357d53243
# SECOND MDOEL: 6dc9dc55739c7c6471fc095f4f1fd43a
# THIRD BEST MODEL: a8fb0d0787a6e359e8fceae8b3d3b6e0
# FOURTH: 144ad3072484aa023e76c7410f99c65a


# det_l is a list of dictionaries, as the detections are split among characters
def merge_and_supress_detections_split(det_l: list[tuple[dict[str, list[list]], str]]):
    # I know the name sucks, split in the sense that split by character
    # Merged in the sense that you merge the detections of multiple classifiers
    # But they are merged per character
    merged_split_detections = {}

    for char_name in det_l[0][0]:
        char_det_l = []
        for detections, name in det_l:
            char_det_l.append(tuple(detections[char_name] + [name]))
        merged_split_detections[char_name] = merge_and_supress_detections(char_det_l)
        
    return merged_split_detections
            
# Returns all detections, scores and file_names ready to be run with evel_detection
def merge_and_supress_detections(det_l: list[tuple]):
    global facial_detector
    # Merge all detections from the list
    res_tup_det = det_l[0]
    for i in range(1, len(det_l)):
        res_tup_det = merge_detections(res_tup_det, det_l[i])
    
    res_detections, res_scores, res_file_names = [], [], []
    # Non maximal supression on each image
    for detections, scores, file_names in get_img_data(res_tup_det[:3]):
        file_name = os.path.join(facial_detector.params.val_dir, "validare", file_names[0])
        img = cv.imread(file_name, cv.IMREAD_GRAYSCALE)
        # facial_detector._show_det_on_img(detections, scores, img)
        detections, scores, file_names = non_maximal_suppression(detections, scores, file_names, img.shape)
        # facial_detector._show_det_on_img(detections, scores, img)
        
        res_detections.append(detections)
        res_scores.append(scores)
        res_file_names.append(file_names)
       
    # Put everything together
    res_detections = np.concatenate(res_detections)
    res_scores = np.concatenate(res_scores)
    res_file_names = np.concatenate(res_file_names)

    # Hopefully not going insane!
    assert all(res_file_names == res_tup_det[2])

    return res_detections, res_scores, res_file_names, res_tup_det[-1]

# Returns the merged detections, scores, file_names and name from the arguments detection tuples
def merge_detections(det_tup1: tuple, det_tup2: tuple):
    res_name = det_tup1[-1] + "\n" + det_tup2[-1]

    im_data1, im_data2 = get_img_data(det_tup1[:-1]), get_img_data(det_tup2[:-1])
    detections, scores, file_names = [], [], []

    i, j = 0, 0

    # Merge the detections
    while i < len(im_data1) and j < len(im_data2):
        if im_data1[i][2][0] < im_data2[j][2][0]:
            detections.append(im_data1[i][0])
            scores.append(im_data1[i][1])
            file_names.append(im_data1[i][2])
            i += 1
        elif im_data1[i][2][0] > im_data2[j][2][0]:
            detections.append(im_data2[j][0])
            scores.append(im_data2[j][1])
            file_names.append(im_data2[j][2])
            j += 1
        else:
            detections.append(np.concatenate((im_data1[i][0], im_data2[j][0])))
            scores.append(np.concatenate((im_data1[i][1], im_data2[j][1])))
            file_names.append(np.concatenate((im_data1[i][2], im_data2[j][2])))
            i += 1
            j += 1
    
    # Leftovers from first
    while i < len(im_data1):
        detections.append(im_data1[i][0])
        scores.append(im_data1[i][1])
        file_names.append(im_data1[i][2])
        i += 1
    
    # Leftovers from second
    while j < len(im_data2):
        detections.append(im_data2[j][0])
        scores.append(im_data2[j][1])
        file_names.append(im_data2[j][2])
        j += 1

    return np.concatenate(detections), np.concatenate(scores), np.concatenate(file_names), res_name


# Returns a generator containing the detections, scores and file_names of each distinct file
def get_img_data(img_tuple: tuple):
    detections, scores, file_names = img_tuple
    last_ind = 0

    result = []
    for i in range(len(file_names)):
        if file_names[last_ind] != file_names[i]:
            result.append((detections[last_ind:i], scores[last_ind:i], file_names[last_ind:i]))
            last_ind = i
    
    result.append((detections[last_ind:], scores[last_ind:], file_names[last_ind:]))
    return result

def merge_dirs(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            merge_dirs(item, target)
        else:
            shutil.move(str(item), str(target))

    src.rmdir()  # remove empty source dir


# def clear_cache_partial(cache_path):
#     cache_path = Path(cache_path)
#     for item in cache_path.iterdir():
#         if item.is_dir():
#             clear_cache_partial(item)
#         else:
#             rem_img_from_cache(item)

# def is_almost_int(x, eps=1e-6):
#     return abs(x - round(x)) < eps

# def rem_img_from_cache(file_path:Path):
#     file_nr = float(file_path.stem)
    # l_095 = log(file_nr, 0.95)
    # l_085 = log(file_nr, 0.85)
    # if file_nr < 1 and (is_almost_int(l_095) or is_almost_int(l_085)):
    #     os.remove(file_path)
    #     print(f"Removed file {file_path}")
    # if file_nr > 1 and file_nr not in [1.2, 1.3] and abs(file_nr - 1.69) > 1e-4:
    #     os.remove(file_path)
    #     print(f"Removed file {file_path}")

def intersection_over_union(bbox_a, bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou


def non_maximal_suppression(image_detections, image_scores, file_names, image_size):
    """
    Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
    Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
    fi in interiorul celeilalte detectii.
    :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
    :param image_scores: numpy array de dimensiune N
    :param image_size: tuplu, dimensiunea imaginii
    :return: image_detections si image_scores care sunt maximale.
    """

    # xmin, ymin, xmax, ymax
    x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
    y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]

    print(x_out_of_bounds, y_out_of_bounds)
    image_detections[x_out_of_bounds, 2] = image_size[1]
    image_detections[y_out_of_bounds, 3] = image_size[0]

    sorted_indices = np.flipud(np.argsort(image_scores))
    sorted_image_detections = image_detections[sorted_indices]
    sorted_scores = image_scores[sorted_indices]
    sorted_file_names = file_names[sorted_indices]

    is_maximal = np.ones(len(image_detections)).astype(bool)
    iou_threshold = 0.3
    for i in range(len(sorted_image_detections) - 1):
        if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
            for j in range(i + 1, len(sorted_image_detections)):
                if is_maximal[j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                    if intersection_over_union(sorted_image_detections[i],sorted_image_detections[j]) > iou_threshold:is_maximal[j] = False
                    else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                        c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                        c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                        if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                            is_maximal[j] = False
    return sorted_image_detections[is_maximal], sorted_scores[is_maximal], sorted_file_names[is_maximal]


def print_dict_size(dic):
    for k, v in dic.items():
        print(f"{k}: {len(v)}")
    print()

def train_detector(detector: FacialDetector):
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

def run_detector(detector: FacialDetector, all_detections, all_files):
    # Test model
    print("Running detector...")
    start_time = timeit.default_timer()
    detections = detector.run(all_detections, all_files)
    end_time = timeit.default_timer()
    print(f"Running detector took: {end_time - start_time} sec!")

    return detections

def normalize_scores(scores: np.ndarray):
    norm = (scores - scores.mean()) / scores.std()
    return norm

def run_project():
    facial_detector: FacialDetector = FacialDetector(Parameters())

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
    best_model_det_dir = os.path.join(facial_detector.params.dir_save_files, "merge_results", "3fa93ef7c172591e58699e9ad51a72ec", "data")

    all_det, all_scores, all_files = facial_detector.fetch_all_detections(best_model_det_dir)
    print(f"Fetched all {len(all_det)} detections!")

    # Run the classifier on the detections
    detections = facial_detector.run(all_det, all_files)

    # Set up files and dirs
    merged_fd_name = "_".join(best_model_det_dir.split('\\')[-3:-1])
    merged_fd_name_hash = hashlib.md5(merged_fd_name.encode()).hexdigest()
    merge_dir = os.path.join(facial_detector.params.dir_save_files, "merge_results_classifier", merged_fd_name_hash)

    # to more easily identify the model(s) used
    merge_info_file = os.path.join(merge_dir, "info.txt")
    os.makedirs(merge_dir, exist_ok=True)
    open(merge_info_file, 'w').write(merged_fd_name)

    # Save detections
    facial_detector.save_detections(detections, merge_dir)

    # Write additional information and evaluate detections
    facial_detector.eval_detections_split(detections, merge_dir)
