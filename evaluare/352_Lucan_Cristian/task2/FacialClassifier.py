from Parameters import *
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pdb
import pickle
import ntpath
from copy import deepcopy
import timeit
from skimage.feature import hog, local_binary_pattern
from skimage.exposure import equalize_adapthist
from joblib import Parallel, delayed
from pathlib import Path

def unzip3(l):
    res1, res2, res3 = [], [], []

    for e1, e2, e3 in l:
        res1.append(e1)
        res2.append(e2)
        res3.append(e3)

    return res1, res2, res3
    
def recursive_len_of_dict(dic: dict[str, list]):
    return sum([len(l) for l in dic.values()])

# TODO: Add the possibility to re-do only the negative descriptors
# Transform the dictionary into a list
def merge_dict(dic: dict[str, np.ndarray]):
    return np.concatenate([v for k, v in dic.items()])

# Saves the value at save_path/dic_key
def save_dictionary(save_path: str, dic: dict[str, np.ndarray]):
    for file_name in dic:
        np.save(os.path.join(save_path, file_name), dic[file_name])

def get_bbox_area(bbox: list):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

class FacialClassifer:
    detection_size = 4

    def __init__(self, params:Parameters):
        self.params = params
        self.best_model = None
        self.bbox_len_list = [] # Keep track of all detection bbox areas for statistical purposes
        self.char_label_dict = {"daphne": 0, "fred": 1, "shaggy": 2, "velma": 3, "unknown": 4}
        self.label_char_dict = {0: "daphne", 1: "fred", 2: "shaggy", 3: "velma", 4: "unknown"}

    # Returns two numpy arrays, one containing all the descriptors, the other with labels
    def get_merged_data(self, split_train_data: dict[str, np.ndarray]):
        desc = []
        labels = []

        for char_name, char_desc in split_train_data.items():
            desc.append(char_desc)
            labels.append(np.full(len(char_desc), self.char_label_dict[char_name]))
        
        return np.concatenate(desc), np.concatenate(labels)
    
    # Returns the positive descriptors of each character as a dictionary
    def get_train_desc(self) -> dict[str, np.ndarray]:
        # Descriptors present, just fetch them
        all_pos_desc_split = self._fetch_desc()
        if all_pos_desc_split:
            print("Fetched descriptor dictionary!")
            return all_pos_desc_split

        print(f"Descriptors not found, computing them...")
        all_pos_desc_split = {}

        # Construct and save postive and negative descriptors
        for file_name in os.listdir(self.params.train_dir):
            file_name = os.path.join(self.params.train_dir, file_name)
            if os.path.isfile(file_name) and ("annotations" in file_name or "gt_validare" in file_name):
                print(f"Handling annotation file: {file_name}")
                pos_desc = self._handle_ann_file(file_name)
                self._add_pos_desc(all_pos_desc_split, pos_desc)

        # Save dictionary and list
        save_dictionary(self.params.pos_desc_dir, all_pos_desc_split)
        
        return all_pos_desc_split 

    # Reads, parses the annotation file, constructs the positive and negative descriptors
    # Returns a dictionary and a list representing the descriptors
    # Dictionary has key:char_name, val:pos_desc
    def _handle_ann_file(self, file_name: str):
        # Load the the data
        ground_truth_file = np.loadtxt(file_name, dtype='str')
        gt_file_names = np.array(ground_truth_file[:, 0])
        gt_detections = np.array(ground_truth_file[:, 1:FacialClassifer.detection_size + 1], int)
        gt_char_names = np.array(ground_truth_file[:, FacialClassifer.detection_size + 1])

        all_pos_desc = {} #key - char_name, val - list of descriptors
        last_ind = 0
        char_name = file_name.split("\\")[-1].split("_")[0]
        img_dir = os.path.join(self.params.train_dir, char_name)

        # Iterate through files, get descriptors for distinct files and add them to the bigger list/dict
        for i in range(1, len(gt_file_names)):
            if gt_file_names[i] != gt_file_names[last_ind]:
                img_f_name = os.path.join(img_dir, gt_file_names[last_ind])
                print(f"Handling image file {img_f_name}")

                # Get positve descriptors for image
                img = cv.imread(img_f_name, cv.IMREAD_GRAYSCALE)
                detections, char_names = gt_detections[last_ind:i], gt_char_names[last_ind:i]
                
                pos_desc = self._get_pos_desc(img, detections, char_names)
                self._add_pos_desc(all_pos_desc, pos_desc)

                last_ind = i
        
        return all_pos_desc

    def get_features(self, img, feature_vector:bool):
        features = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                       cells_per_block=(self.params.dim_block, self.params.dim_block), feature_vector=feature_vector)
        return features
        
    # Returns a dictionary, char_name:descriptors
    def _fetch_desc(self) -> dict[str, np.ndarray]:
        pos_desc_dir = Path(self.params.pos_desc_dir)
        pos_desc_dic = {}

        for file_name in pos_desc_dir.iterdir():
            char_name = file_name.stem
            if char_name in self.char_label_dict:
                pos_desc_dic[char_name] = np.load(file_name)
        
        return pos_desc_dic

    # Returns the positive descriptors for the detections in img as a dictionary
    # key: char_name, value: pos_desc
    def _get_pos_desc(self, img: np.ndarray, detections: np.ndarray, char_names: np.ndarray):
        pos_desc_split = {name:[] for name in char_names}

        for i in range(len(detections)):
            bbox, char_name = detections[i], char_names[i]
            
            # Only process faces withing a given area range
            area_l = np.sqrt(get_bbox_area(bbox))
            if area_l < self.params.dim_window_lower or area_l > self.params.dim_window_upper:
                continue

            # Extract face and resize it to window dimensions
            face = cv.resize(img[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy(), (self.params.dim_window, self.params.dim_window), interpolation=cv.INTER_AREA)
            # Extract hog desc
            face_features = self.get_features(face, True)
            # Add it to the dictionary
            pos_desc_split[char_name].append(face_features)

            # The same for flipped face
            if self.params.use_flip_images:
                face_features = self.get_features(face, True)
                pos_desc_split[char_name].append(face_features)
        
        pos_desc_split = {name:np.array(pos_desc_split[name]) for name in pos_desc_split}
        return pos_desc_split

    # Adds the positive desc in pos_desc to all_pos_desc, changing it
    def _add_pos_desc(self, all_pos_desc:dict[str, np.ndarray], pos_desc: dict[str, np.ndarray]):
        pos_desc = {k:v for k, v in pos_desc.items() if len(v) > 0}
        for char_name in pos_desc:
            if char_name in all_pos_desc:
                all_pos_desc[char_name] = np.concatenate((all_pos_desc[char_name], pos_desc[char_name]))
            else:
                all_pos_desc[char_name] = pos_desc[char_name]

    
    # Returns the linear SVC with the most accuracy, each svc differs by the C param
    def _train(self, training_examples, train_labels):
        print("Training model...")
        best_accuracy = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2, 10 ** -1, 10 ** 0]
        start_time = timeit.default_timer()
        for c in Cs:
            model = LinearSVC(C=c, class_weight=self.params.class_weights, max_iter=self.params.SVC_iter)
            model.fit(training_examples, train_labels)

            acc = model.score(training_examples, train_labels)
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = deepcopy(model)

        end_time = timeit.default_timer()
        print(f"Training duration: {end_time - start_time} seconds")
        print(f"Accuracy: {best_accuracy}")
        return best_model, best_accuracy

    def set_model(self):
        print(f"Fetched model: {self.params.class_model_file}")
        self.best_model = pickle.load(open(self.params.class_model_file, 'rb'))

    def save_model(self):
        print(f"Saved model: {self.params.class_model_file}")
        pickle.dump(self.best_model, open(self.params.class_model_file, 'wb'))

    # Trains and saves a linear classifier
    def train_classifier(self, training_examples, train_labels):
        if os.path.exists(self.params.class_model_file):
            self.set_model()
            return
        
        self.best_model, _ = self._train(training_examples, train_labels)
        self.save_model()

    # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
    # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
    def plot_train(self, title, training_examples, train_labels):
        scores = self.best_model.decision_function(training_examples)
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]

        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(positive_scores)))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Training sample count')
        plt.ylabel('Classifier score')
        plt.title(title)
        plt.legend(['Positive examples score', '0', 'Negative examples scores'])
        plt.savefig(os.path.join(self.params.model_dir, f"{title}.png"))
        plt.close()

    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou
    
    def fetch_all_detections(self, merge_dir:str|None=None):
        if merge_dir is None:
            det_dir = self.params.test_res_data_dir_all
        else:
            det_dir = merge_dir

        detections = np.load(os.path.join(det_dir, "detections_all_faces.npy"))
        scores = np.load(os.path.join(det_dir, "scores_all_faces.npy"))
        file_names = np.load(os.path.join(det_dir, "file_names_all_faces.npy"))
        
        return detections, scores, file_names
    
    def save_detections_char(self, char_test_res_data_dir: str, detections: np.ndarray, scores: np.ndarray, file_names: np.ndarray):
        if not os.path.exists(char_test_res_data_dir):
            os.makedirs(char_test_res_data_dir)

        np.save(os.path.join(char_test_res_data_dir, "detections.npy"), detections)
        np.save(os.path.join(char_test_res_data_dir, "scores.npy"), scores)
        np.save(os.path.join(char_test_res_data_dir, "file_names.npy"), file_names)

    def save_detections(self, detections: dict[str, list[np.ndarray]], merge_dir:str|None = None):
        det_save_dir = self.params.test_res_data_dir_split
        if merge_dir is not None:
            det_save_dir = merge_dir

        for char_name in detections:
            char_test_res_data_dir = os.path.join(det_save_dir, char_name)
            self.save_detections_char(char_test_res_data_dir, detections[char_name][0], detections[char_name][1], detections[char_name][2])

    # Returns the label and score of the face
    def predict(self, faces_desc: np.ndarray):
        all_scores = self.best_model.decision_function(faces_desc)
        labels = self.best_model.classes_

        max_scores = [[] for _ in range(len(all_scores))]

        # Keep only the max score of each detection together with label
        for i in range(len(all_scores)):
            ind = np.argmax(all_scores[i])
            max_scores[i] = [all_scores[i][ind], labels[ind], i]
        
        # No fancy verification for a character appearing twice
        if self.params.two_faced:
            scores, labels, _ = unzip3(max_scores)
            return labels, scores, all_scores
        
        # Sort based on scores
        max_scores = sorted(max_scores, key=lambda x: -x[0])
        available_labels = set(labels)

        scores = []
        labels = []

        # Faces can't appear twice! If a face was labeled twice set it to unknown!
        for i, elem in enumerate(max_scores):
            if elem[1] in available_labels:
                available_labels.remove(elem[1])
            else:
                max_scores[i][1] = self.char_label_dict["unknown"]

        max_scores = sorted(max_scores, key=lambda x: x[-1])
        scores, labels, _ = unzip3(max_scores)
        
        return labels, scores, all_scores
    
    # Returns a dictionary where key : char_name and value is a list of list where
    # dict_value[0] - detections(bboxes), dict_value[1] - scores, dict_value[2] - file_names
    def run(self, all_detections: np.ndarray, all_files: np.ndarray):
        print("Running test/validation...")

        # dict_value[0] - detections(bboxes), dict_value[1] - scores, dict_value[2] - file_names
        detections_split = {char_name: [[], [], []] for char_name in self.char_label_dict}

        # For each detection from the face/non-face model extract the patch from the image
        # Run hog and call predict
        last_ind = 0
        for i in range(len(all_detections)):
            # New image
            if all_files[last_ind] != all_files[i]:
                img = cv.imread(os.path.join(self.params.val_dir, "validare", all_files[last_ind]), cv.IMREAD_GRAYSCALE)
                all_descriptors = []
                for det in all_detections[last_ind:i]:
                    patch = img[det[1]:det[3], det[0]:det[2]]
                    patch = cv.resize(patch, (self.params.dim_window, self.params.dim_window))
                    all_descriptors.append(self.get_features(patch, True))
                labels, scores, all_scores = self.predict(np.array(all_descriptors))

                # self._show_det_on_img(all_detections[last_ind:i], scores, labels, img)

                # Add classifications of this image
                for j, label in enumerate(labels):
                    label = self.label_char_dict[label]
                    detections_split[label][0].append(all_detections[last_ind + j])
                    detections_split[label][1].append(scores[j])     
                    detections_split[label][2].append(all_files[last_ind + j])

                last_ind = i

        # Last file
        img = cv.imread(os.path.join(self.params.val_dir, "validare", all_files[last_ind]), cv.IMREAD_GRAYSCALE)        
        all_descriptors = np.array([self.get_features(cv.resize(img[det[1]:det[3], det[0]:det[2]], (self.params.dim_window, self.params.dim_window)), True) for det in all_detections[last_ind:]])
        labels, scores, all_scores = self.predict(all_descriptors)

        # self._show_det_on_img(all_detections[last_ind:], scores, labels, img)

        # Add classifications of this image
        for j, label in enumerate(labels):
            label = self.label_char_dict[label]
            detections_split[label][0].append(all_detections[last_ind + j])
            detections_split[label][1].append(scores[j])     
            detections_split[label][2].append(all_files[last_ind + j])

        # Make them numpy arrays
        for label in detections_split:
            detections_split[label][0] = np.array(detections_split[label][0])
            detections_split[label][1] = np.array(detections_split[label][1])
            detections_split[label][2] = np.array(detections_split[label][2])

        return detections_split

    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def eval_detections_character(self, detections, scores, file_names, character, merge_dir: str|None = None):
        # Set save dir
        save_dir = self.params.test_res_data_dir_split
        if merge_dir is not None:
            save_dir = merge_dir

        gt_path = os.path.join(self.params.val_dir, f"task2_{character}_gt_validare.txt")

        ground_truth_file = np.loadtxt(gt_path, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:], int)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(character + ' faces: average precision %.3f' % average_precision)
        plt.savefig(os.path.join(save_dir, 'precizie_medie_' + character + '.png'))
        plt.close()
    
    def eval_detections_split(self, detections, merge_dir: str|None = None):
        for char in self.char_label_dict:
            if char != "unknown":
                self.eval_detections_character(detections[char][0], detections[char][1], detections[char][2], char, merge_dir)

    # Returns a string that identifies all the relevant model parameters
    def get_name(self):
        return "_".join(self.params.class_model_dir.split("\\")[-6:])

    def _show_det_on_img(self, detections, scores, labels, img):
        img_cp = img.copy()

        for i, detection in enumerate(detections):
            cv.rectangle(img_cp, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), thickness=1)
            
            cv.putText(img_cp, f'score_{labels[i]}:' + str(scores[i])[:4], (detection[0], detection[1] + i * 20),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv.imshow('image', np.uint8(img_cp))
        cv.waitKey(0)
