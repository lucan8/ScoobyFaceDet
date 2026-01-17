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

def recursive_len_of_dict(dic: dict[str, list]):
    return sum([len(l) for l in dic.values()])

# CAREFUL: SVM SCORES DIFFER, DOING NMS WON'T WORK AS USUAL PROBABLY
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

class FacialDetector:
    detection_size = 4

    def __init__(self, params:Parameters):
        self.params = params
        self.best_model = None
        self.bbox_len_list = [] # Keep track of all detection bbox areas for statistical purposes


    # Returns the positive and negative descriptors for the training data as 2 lists
    # TODO: Add to support to get the dictionary as well
    def get_train_desc(self):
        # Descriptors present, just fetch them
        if os.path.exists(self.params.all_pos_desc_file):
            return self._fetch_desc()

        print(f"Descriptors not found, computing them...")
        all_pos_desc_split = {}
        all_neg_desc = []

        # Construct and save postive and negative descriptors
        for file_name in os.listdir(self.params.train_dir):
            file_name = os.path.join(self.params.train_dir, file_name)
            if os.path.isfile(file_name):
                print(f"Handling annotation file: {file_name}")
                pos_desc, neg_desc = self._handle_ann_file(file_name)

                self._add_pos_desc(all_pos_desc_split, pos_desc)
                all_neg_desc.extend(neg_desc)
        
        all_neg_desc = np.array(all_neg_desc)

        # Transform the list in a dictionary
        all_pos_desc_merged = merge_dict(all_pos_desc_split)

        # Save dictionary and list
        save_dictionary(self.params.pos_desc_dir, all_pos_desc_split)
        np.save(self.params.all_pos_desc_file, all_pos_desc_merged)
        np.save(self.params.all_neg_desc_file, all_neg_desc)
        return all_pos_desc_merged, all_neg_desc

    # Reads, parses the annotation file, constructs the positive and negative descriptors
    # Returns a dictionary and a list representing the descriptors
    # Dictionary has key:char_name, val:pos_desc
    def _handle_ann_file(self, file_name: str):
        # Load the the data
        ground_truth_file = np.loadtxt(file_name, dtype='str')
        gt_file_names = np.array(ground_truth_file[:, 0])
        gt_detections = np.array(ground_truth_file[:, 1:FacialDetector.detection_size + 1], int)
        gt_char_names = np.array(ground_truth_file[:, FacialDetector.detection_size + 1])

        all_neg_desc = []
        all_pos_desc = {} #key - char_name, val - list of descriptors
        last_ind = 0
        char_name = file_name.split("\\")[-1].split("_")[0]
        img_dir = os.path.join(self.params.train_dir, char_name)

        # Iterate through files, get descriptors for distinct files and add them to the bigger list/dict
        for i in range(1, len(gt_file_names)):
            if gt_file_names[i] != gt_file_names[last_ind]:
                img_f_name = os.path.join(img_dir, gt_file_names[last_ind])
                print(f"Handling image file {img_f_name}")

                # Get positive and negative desc for image
                img = cv.imread(img_f_name, cv.IMREAD_GRAYSCALE)
                detections, char_names = gt_detections[last_ind:i], gt_char_names[last_ind:i]
                
                pos_desc = self._get_pos_desc(img, detections, char_names)
                self.params.neg_patch_count_per_img = int(recursive_len_of_dict(pos_desc) * self.params.neg_patch_factor)
                neg_desc = self._get_neg_desc(img, detections)
                
                # Add them to bigger dict/list
                self._add_pos_desc(all_pos_desc, pos_desc)
                all_neg_desc.extend(neg_desc)

                last_ind = i
        
        return all_pos_desc, np.array(all_neg_desc)

    def get_lbp_features(self, image, P=8, R=1, feature_vector=True):
        lbp = local_binary_pattern(image, P, R, method="uniform")
        h, w = lbp.shape

        features = []
        for i in range(0, h, self.params.dim_hog_cell):
            row = []
            for j in range(0, w, self.params.dim_hog_cell):
                cell = lbp[i:i+self.params.dim_hog_cell, j:j+self.params.dim_hog_cell]
                hist, _ = np.histogram(
                    cell.ravel(),
                    bins=np.arange(0, P + 3),
                    range=(0, P + 2)
                )
                row.append(hist.astype(np.float64))
            features.append(row)

        features = np.array(features)
        h, w = features.shape[:2]

        block_features = []
        for i in range(0, h - self.params.dim_block + 1, self.params.block_stride):
            row = []
            for j in range(0, w - self.params.dim_block + 1, self.params.block_stride):
                block = features[i:i+self.params.dim_block, j:j+self.params.dim_block].copy()

                block_vec = block.reshape(-1)
                block_vec /= (np.linalg.norm(block_vec) + 1e-6)
                block = block_vec.reshape(block.shape)

                row.append(block)
            block_features.append(row)

        block_features = np.array(block_features)
        if feature_vector:
            block_features = block_features.flatten()

        return block_features


    #TODO: Be careful about feature vector and adding multiple descriptors
    def get_features(self, img, feature_vector:bool):
        # Cut to perfectly fit the cells
        # cell = self.params.dim_hog_cell
        # h_img, w_img = img.shape

        # h_valid = (h_img // cell) * cell
        # w_valid = (w_img // cell) * cell
        # img_valid = img[:h_valid, :w_valid]

        features = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                       cells_per_block=(self.params.dim_block, self.params.dim_block), feature_vector=feature_vector)
        
        if self.params.use_lbp:
            lbp_d = self.get_lbp_features(img, feature_vector=feature_vector)
            features = np.concatenate([features, lbp_d], axis=-1)
           
        return features
        
    # Returns the two lists containing the positive and negative descriptors
    def _fetch_desc(self):
        return np.load(self.params.all_pos_desc_file), np.load(self.params.all_neg_desc_file)
    
    # Eliminating the separation just for testing purposes
    def _is_soft_neg_det(self, ious: np.ndarray):
        return all(ious < self.params.soft_neg_overlap)
    
    # Using all work better by 3%???
    def _is_hard_neg_det(self, ious: np.ndarray):
        return all(ious < self.params.soft_neg_overlap)
    
    # def _is_soft_neg_det(self, ious: np.ndarray):
    #     return all(ious < self.params.soft_neg_overlap)
    
    # # Using all work better by 3%???
    # def _is_hard_neg_det(self, ious: np.ndarray):
    #     return all((self.params.soft_neg_overlap <= ious) & (ious < self.params.overlap))
    
    def _is_hard_pos_det(self, ious: np.ndarray):
        return any(ious > self.params.hard_pos_overlap)
    
    # Returns a list of hog descriptors for random negative patches
    def _get_neg_desc(self, img: np.ndarray, bboxes: np.ndarray):
        H, W = img.shape
        
        soft_sample_count = np.ceil(self.params.neg_patch_count_per_img * self.params.soft_neg_patch_perc)
        hard_sample_count = np.ceil(self.params.neg_patch_count_per_img * (1 - self.params.soft_neg_patch_perc))
        
        hard_neg_patches = []
        soft_neg_patches = []

        max_tries = 1000
        tries = 0

        while tries <= max_tries and (len(soft_neg_patches) < soft_sample_count or len(hard_neg_patches) < hard_sample_count):
            tries += 1

            x = np.random.randint(0, W - self.params.dim_window)
            y = np.random.randint(0, H - self.params.dim_window)

            # Extract patch
            cand = [x, y, x + self.params.dim_window, y + self.params.dim_window]

            # Get patch ious with gt
            cand_ious = np.array([self.intersection_over_union(cand, bbox) for bbox in bboxes])

            # Soft negative
            if len(soft_neg_patches) < soft_sample_count and self._is_soft_neg_det(cand_ious):
                patch = img[cand[1]:cand[3], cand[0]:cand[2]]
                patch = self.get_features(patch, True)
                soft_neg_patches.append(patch)
            # Hard negative
            elif len(hard_neg_patches) < hard_sample_count and self._is_hard_neg_det(cand_ious):
                patch = img[cand[1]:cand[3], cand[0]:cand[2]]
                patch = self.get_features(patch, True)
                hard_neg_patches.append(patch)
                
        soft_neg_patches.extend(hard_neg_patches)
        return np.array(soft_neg_patches)

    def set_bbox_len_list(self):
        if os.path.exists(self.params.bbox_len_list_file):
            self.bbox_len_list = np.load(self.params.bbox_len_list_file)
            return
        # For each annotation file in train dir read the bboxes of the detections and add their length to bbox_len_list
        for file_name in os.listdir(self.params.train_dir):
            file_name = os.path.join(self.params.train_dir, file_name)

            if os.path.isfile(file_name):
                ground_truth_file = np.loadtxt(file_name, dtype='str')
                gt_detections = np.array(ground_truth_file[:, 1:FacialDetector.detection_size + 1], int)

                for gt_det in gt_detections:
                    bbox_len = int(np.sqrt((gt_det[2] - gt_det[0]) * (gt_det[3] - gt_det[1])))
                    self.bbox_len_list.append(bbox_len)
        np.save(self.params.bbox_len_list_file, self.bbox_len_list)


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


    # Returns a list containing all the hard negative descritors
    def _get_hard_desc(self):
        all_neg_desc = []
        all_pos_desc = []

        for file_name in os.listdir(self.params.train_dir):
            file_name = os.path.join(self.params.train_dir, file_name)
            if os.path.isfile(file_name):
                print(f"Hard mining for {file_name}...")
                
                neg_desc, pos_desc = self._get_hard_desc_for_char(file_name)
                all_neg_desc.extend(neg_desc)
                all_pos_desc.extend(pos_desc)

        return np.array(all_neg_desc), np.array(all_pos_desc)

    # Returns the hard neg desc for the characther with arg as it's annotation file
    def _get_hard_desc_for_char(self, file_name: str):
        # Load the the data
        ground_truth_file = np.loadtxt(file_name, dtype='str')
        gt_file_names = np.array(ground_truth_file[:, 0])

        # Keep the indexes of only distinct files
        last_ind = 0
        distinct_gt_f_names_ind = []
        for i in range(1, len(gt_file_names)):
            if gt_file_names[last_ind] != gt_file_names[i]:
                distinct_gt_f_names_ind.append(np.array([last_ind, i]))
                last_ind = i
        # Don't forget about last image
        distinct_gt_f_names_ind.append(np.array([last_ind, len(gt_file_names)]))
        distinct_gt_f_names_ind = np.array(distinct_gt_f_names_ind)

        gt_detections = np.array(ground_truth_file[:, 1:FacialDetector.detection_size + 1], int)

        all_neg_desc = []
        all_pos_desc = []
        char_name = file_name.split("\\")[-1].split("_")[0]
        img_dir = os.path.join(self.params.train_dir, char_name)

        # Generate random indexes
        chosen_ind = np.random.choice(np.array(range(len(distinct_gt_f_names_ind))), self.params.hard_neg_img_count_per_char, replace=False)
        chosen_ind = distinct_gt_f_names_ind[chosen_ind]
        
        for ind_pair in chosen_ind:
            start_ind, end_ind = ind_pair[0], ind_pair[1]
            img_f_name = os.path.join(img_dir, gt_file_names[start_ind])
            detections = gt_detections[start_ind:end_ind]

            # Get the negative descriptor for the image
            start_time = timeit.default_timer()
            neg_desc, pos_desc = self._get_both_hard_desc_for_img(img_f_name, detections)
            end_time = timeit.default_timer()
            print(f"Hard mined {img_f_name} in {end_time - start_time} sec!")
            
            all_neg_desc.extend(neg_desc)
            all_pos_desc.extend(pos_desc)
        
        return np.array(all_neg_desc), np.array(all_pos_desc)
    
    
    # Run the detector on img and returns a list of hard negative descriptors
    # TODO: Improvement: Choose detections based on characters
    def _get_hard_desc_for_img(self, img_name: str, gt_detections: np.ndarray, negative: bool):
        # Make a softer predict threshold to get more false predictions
        init_thr = self.params.threshold
        if negative:
            self.params.threshold = 0
            hard_filter_func = self._is_hard_neg_det
            ind_range = (-self.params.hard_count_per_img, None)
        else: 
            self.params.threshold = -1000
            hard_filter_func = self._is_hard_pos_det
            ind_range = (0, self.params.hard_count_per_img)

        # Get all detections on this image
        detections, scores, hog_desc = self.predict(img_name, True)
        
        # Keep only indexes having detections diff from gt
        detections_ious = np.array([np.array([self.intersection_over_union(detection, bb) for bb in gt_detections]) for detection in detections])
        ind = [index for index, ious in enumerate(detections_ious) if hard_filter_func(ious)]

        # Keep track of the changes
        scores = scores[ind]
        det = detections[ind]
        hog_desc = hog_desc[ind]

        # Keep only the top max_false_pos and add
        ind = np.argsort(scores)[ind_range[0]:ind_range[1]]

        # Reset threshold
        self.params.threshold = init_thr

        # Just want the negatives for now
        return hog_desc[ind]
        # return neg_hog_desc[neg_ind], pos_hog_desc[pos_ind]

    # Run the detector on img and returns a list of both hard negative and positive descriptors
    # If self.params.hard_pos_overlap == 0 only returns the negatives!
    def _get_both_hard_desc_for_img(self, img_name: str, gt_detections: np.ndarray):
        # Only want the negatives
        if self.params.hard_pos_overlap == 0:
            return self._get_hard_desc_for_img(img_name, gt_detections, True), np.array([])
        
        # Make a softer predict threshold to get more false predictions
        init_thr = self.params.threshold
        self.params.threshold = -1000

        # Get all detections on this image
        detections, scores, hog_desc = self.predict(img_name, True)
        
        # Keep only indexes having detections diff from gt
        detections_ious = np.array([np.array([self.intersection_over_union(detection, bb) for bb in gt_detections]) for detection in detections])
        neg_ind = [index for index, ious in enumerate(detections_ious) if self._is_hard_neg_det(ious)]
        pos_ind = [index for index, ious in enumerate(detections_ious) if self._is_hard_pos_det(ious)]

        # Keep track of the changes
        neg_scores, pos_scores = scores[neg_ind], scores[pos_ind]
        neg_det, pos_det = detections[neg_ind], detections[pos_ind]
        neg_hog_desc, pos_hog_desc = hog_desc[neg_ind], hog_desc[pos_ind]

        # Keep only the top max_false_pos and add
        neg_ind = np.argsort(neg_scores)[-self.params.hard_count_per_img:]
        pos_ind = np.argsort(pos_scores)[:self.params.hard_count_per_img]

        # img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
        # # Show detections
        # self._show_det_on_img(neg_det[neg_ind], neg_scores[neg_ind], img)
        # self._show_det_on_img(pos_det[pos_ind], pos_scores[pos_ind], img)

        # Reset threshold
        self.params.threshold = init_thr

        return neg_hog_desc[neg_ind], pos_hog_desc[pos_ind]

    
    # Returns the linear SVC with the most accuracy, each svc differs by the C param
    def _train(self, training_examples, train_labels):
        best_accuracy = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2, 10 ** -1, 10 ** 0]
        start_time = timeit.default_timer()
        for c in Cs:
            model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)

            acc = model.score(training_examples, train_labels)
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = deepcopy(model)

        end_time = timeit.default_timer()
        print(f"Training duration: {end_time - start_time} seconds")
        return best_model, best_accuracy
    
    # Returns the negative and positive descriptors of all models that precede this one + number of done iterations
    # Also sets best_model to the newest model
    def _fetch_prev_mined_data(self):
        # Information about the model the precedes this one
        hard_mine_params = self.params.model_dir.split("\\")[-1].split("_")
        it_count_done, others = int(hard_mine_params[1]), hard_mine_params[2:]

        neg_desc = []
        pos_desc = []
        it_count_done = 0

        # Look for model 0 separately
        model0_file = os.path.join(self.params.run_dir, "model_0", "model")
        if os.path.exists(model0_file):
            print("Fetched model0!")
            self.best_model = pickle.load(open(model0_file, 'rb'))

        # Get data from older iterations
        for i in range(1, self.params.hard_neg_mining_it_count):
            model_dir = os.path.join(self.params.run_dir, f"model_{i}_{"_".join(others)}")
            hard_min_data_dir = os.path.join(model_dir, "hard_min_data")

            # Load positive and negative descriptors + model
            if os.path.exists(hard_min_data_dir):
                print(f"Fetched predecessor model {i}!")
                neg_desc.extend(np.load(os.path.join(hard_min_data_dir, "neg_desc.npy")))
                pos_desc.extend(np.load(os.path.join(hard_min_data_dir, "pos_desc.npy")))
                self.best_model = pickle.load(open(os.path.join(model_dir, "model"), 'rb'))
                it_count_done = i
            else:
                break
        
        return np.array(neg_desc), np.array(pos_desc), it_count_done

    def set_model(self):
        print(f"Fetched model: {self.params.model_file}")
        self.best_model = pickle.load(open(self.params.model_file, 'rb'))

    # Trains and saves a linear classifier
    def train_classifier(self, training_examples, train_labels):
        if os.path.exists(self.params.model_file):
            self.best_model = pickle.load(open(self.params.model_file, 'rb'))
            return

        # Fetch prev mined data and set to older model
        hard_neg_desc, hard_pos_desc, it_count_done = self._fetch_prev_mined_data()
        print(f"Additional hard mined desc: {len(hard_pos_desc)}, {len(hard_neg_desc)}")

        # Append to already existing data
        if len(hard_neg_desc):
            training_examples = np.vstack((training_examples, hard_neg_desc))
            train_labels = np.concatenate((train_labels, np.zeros(len(hard_neg_desc))))
        
        if len(hard_pos_desc):
            training_examples = np.vstack((training_examples, hard_pos_desc))
            train_labels = np.concatenate((train_labels, np.ones(len(hard_pos_desc))))
        
        print(f"Starting training desc size: {len(training_examples)}")
        # Train from scratch if no older model exists
        if self.best_model is None:
            self.params.reset_model_dir(0)
            self._train_and_eval(training_examples, train_labels)
        
        # Hard negative mining
        for i in range(it_count_done + 1, self.params.hard_neg_mining_it_count + 1):
            self.params.hard_pos_overlap = self.params.hard_pos_overlap_base + (i - 1) * self.params.hard_pos_overlap_step
            print(f"Hard negative mining iteration {i}...")
            
            # Get descriptors for hard negatives
            start_time = timeit.default_timer()
            hard_neg_desc, hard_pos_desc = self._get_hard_desc()
            end_time = timeit.default_timer()

            print(f"Duration: {end_time - start_time} sec")
            print(f"Hard negative descriptor count for iteration {i}: {len(hard_neg_desc)}")
            print(f"Hard positive descriptor count for iteration {i}: {len(hard_pos_desc)}")

            # Reset model dirs to save the data of this iteration
            self.params.reset_model_dir(i)
            
            # Reshape and add hard examples(also save them)
            if len(hard_neg_desc):
                np.save(self.params.hard_min_neg_desc_file, hard_neg_desc)
                hard_neg_desc = np.array(hard_neg_desc).reshape(len(hard_neg_desc), -1)
                training_examples = np.vstack((training_examples, hard_neg_desc))
                train_labels = np.concatenate((train_labels, np.zeros(len(hard_neg_desc))))
            if len(hard_pos_desc):
                np.save(self.params.hard_min_pos_desc_file, hard_pos_desc)
                hard_pos_desc = np.array(hard_pos_desc).reshape(len(hard_pos_desc), -1)
                training_examples = np.vstack((training_examples, hard_pos_desc))
                train_labels = np.concatenate((train_labels, np.ones(len(hard_pos_desc))))
            
            self._train_and_eval(training_examples, train_labels)
    
    def _train_and_eval(self, training_examples, train_labels):
        # Retrain
        print("Retraining model...")
        self.best_model, _ = self._train(training_examples, train_labels)

        # Save the classifier
        pickle.dump(self.best_model, open(self.params.model_file, 'wb'))
        self.plot_train("train_results", training_examples, train_labels)

        detections, scores, files, _ = self.run()
        self.eval_detections(detections, scores, files)
        self.eval_detections(detections, scores, files, True)

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

    def non_maximal_suppression(self, image_detections, image_scores, image_hog_desc, image_size):
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

        # print(x_out_of_bounds, y_out_of_bounds)
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]

        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]
        sorted_hog_desc = image_hog_desc[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                        if self.intersection_over_union(sorted_image_detections[i],sorted_image_detections[j]) > iou_threshold:is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False
        
        detections, scores, descriptors = sorted_image_detections[is_maximal], sorted_scores[is_maximal], sorted_hog_desc[is_maximal]
        assert len(detections) == len(scores) and len(detections) == len(descriptors)

        return detections, scores, descriptors

    # Returns all detections found in img_name for the given resize
    def process_scale(self, img_init: np.ndarray, cache_dir: str, scale_step: float, scale: float):
        detections, scores, desc  = [], [], []

        w = self.best_model.coef_.T
        bias = self.best_model.intercept_[0]

        cache_file = os.path.join(cache_dir, f"{scale}.npy")

        img = cv.resize(img_init, None, fx=scale_step, fy=scale_step, interpolation=cv.INTER_LINEAR)
        # Try loading descriptors from cache
        if os.path.exists(cache_file):
            hog_descriptors = np.load(cache_file)
            # print("loaded from cache")
        else: # Or create and save them
            hog_descriptors = self.get_features(img, False)
            if self.params.use_cache: 
                os.makedirs(cache_dir, exist_ok=True)
                np.save(cache_file, hog_descriptors)

        # Each row in this reshaped array is one window's feature vector for your SVM
        num_rows, num_cols = hog_descriptors.shape[:2]
        num_cell_in_template = self.params.dim_window // self.params.dim_hog_cell - self.params.dim_block + 1
        
        # Slide window over hogged resized image
        for y in range(0, num_rows - num_cell_in_template):
            for x in range(0, num_cols - num_cell_in_template):
                descr = hog_descriptors[y:y + num_cell_in_template, x:x + num_cell_in_template].flatten()
                score = np.dot(descr, w)[0] + bias
                # Get bounding box for positive detections
                if score > self.params.threshold:
                    x_min = int((x * self.params.dim_hog_cell) / scale)
                    y_min = int((y * self.params.dim_hog_cell) / scale)
                    x_max = int((x * self.params.dim_hog_cell + self.params.dim_window) / scale)
                    y_max = int((y * self.params.dim_hog_cell + self.params.dim_window) / scale)
                    # Might go out of bounds with the window
                    if x_min < img_init.shape[1] and y_min < img_init.shape[0]:
                        detections.append([x_min, y_min, x_max, y_max])
                        scores.append(score)
                        desc.append(descr)

        return detections, scores, desc, img
    
    def _merge_joblib_results(self, results:list[tuple]):
        all_detections = []
        all_scores = []
        all_desc = []

        for dets, scores, desc in results:
            all_detections.extend(dets)
            all_scores.extend(scores)
            all_desc.extend(desc)
        
        return np.array(all_detections), np.array(all_scores), np.array(all_desc)
    
    def _process_scales(self, img: np.ndarray, scale_step:float, cache_dir: str, thr: float):
        results = []
        scale = 1
        window_dim = self.params.dim_window
        if window_dim > thr:
            mult = -1
        else:
            mult = 1
        diff_thr = 2
        results = []
        # Process scales until threshold is hit
        while mult * (thr - window_dim) > diff_thr:
            window_dim /= scale_step
            scale *= scale_step
            result = self.process_scale(img, cache_dir, scale_step, scale)
            # Update the image with the scaled one
            img = result[-1]
            # Add the detections, scores and descriptors to results and update the scale
            results.append(result[:3])
            # print(f"Processed scale:{scale}")

        return results

    # Returns detections, scores and file_names
    def fetch_detections(self):
        detections = np.load(self.params.test_res_det_file)
        scores = np.load(self.params.test_res_scores_file)
        file_names = np.load(self.params.test_res_file_names_file)
        hog_desc = np.load(self.params.test_res_hog_file)

        return detections, scores, file_names, hog_desc
    
    # Save detections to file
    def save_detections(self, detections: np.ndarray, scores: np.ndarray, file_names: np.ndarray, hog_desc: np.ndarray = None):
        np.save(self.params.test_res_det_file, detections)
        np.save(self.params.test_res_scores_file, scores)
        np.save(self.params.test_res_file_names_file, file_names)
        if hog_desc is not None:
            np.save(self.params.test_res_hog_file, hog_desc)

    # Returns the bounding boxes, scores and hog descriptors of the found faces, each as a list
    def predict(self, img_name: str, hard_mining=False):
        # Extract the number and char name for constructing the cache path
        img_name_split = img_name.split("\\")
        img_nr, char_name = img_name_split[-1].split(".")[0], img_name_split[-2]

        img_resizes = self.params.image_resizes

        # Adjust window sizes and cache directory based on what we are doing
        if hard_mining:
            if not self.params.use_all_hard_min_resizes: # Keep only the middle ones
                mid = len(self.params.image_resizes) // 2
                mid_of_mid = mid // 2
                img_resizes = img_resizes[mid-mid_of_mid:mid+mid_of_mid]
            cache_dir = os.path.join(self.params.train_cache, char_name, img_nr)
        else:
            cache_dir = os.path.join(self.params.val_cache, img_nr)

        # Read img
        img_init = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
        img = img_init
        results = [self.process_scale(img, cache_dir, 1, 1)[:3]]

        # Upscaling
        results.extend(self._process_scales(img, self.params.up_step, cache_dir, self.params.dim_window_lower))
        
        # Downscaling
        results.extend(self._process_scales(img, self.params.down_step, cache_dir, self.params.dim_window_upper))

        # Merge results
        image_detections, image_scores, image_desc = self._merge_joblib_results(results)
        
        # Run non maximal suppression
        if not hard_mining:
            if len(image_scores) > 0:
                image_detections, image_scores, image_desc = self.non_maximal_suppression(
                                                                            image_detections,
                                                                            image_scores,
                                                                            image_desc,
                                                                            img_init.shape)
            
        
        # self._show_det_on_img(image_detections, image_scores, img_init)
        return image_detections, image_scores, image_desc

    def run(self, validation=True):
        """
        Aceasta functie returneaza toate detectiile ( = ferestre) pentru toate imaginile din self.params.dir_test_examples
        Directorul cu numele self.params.dir_test_examples contine imagini ce
        pot sau nu contine fete. Aceasta functie ar trebui sa detecteze fete atat pe setul de
        date MIT+CMU dar si pentru alte imagini
        Functia 'non_maximal_suppression' suprimeaza detectii care se suprapun (protocolul de evaluare considera o detectie duplicata ca fiind falsa)
        Suprimarea non-maximelor se realizeaza pe pentru fiecare imagine.
        :return:
        detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
        detections[i, :] = [x_min, y_min, x_max, y_max]
        scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
        file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
        (doar numele, nu toata calea).
        """

        print("Running test/validation...")
        if validation:
            test_images_path = os.path.join(self.params.val_dir, "validare", "*.jpg")
        else:
            test_images_path = os.path.join(self.params.test_dir, '*.jpg')
        
        # Don't rerun if not needed
        if os.path.exists(self.params.test_res_hog_file):
            print("Fetched detections! No more running!")
            return self.fetch_detections()
        
        test_files = glob.glob(test_images_path)

        detections = None
        scores = np.array([])
        file_names = np.array([])
        descriptors = None

        num_test_images = len(test_files)
        
        for i in range(num_test_images):
            start_time = timeit.default_timer()

            print('Processing image %d/%d..' % (i, num_test_images))
            
            image_detections, image_scores, image_desc = self.predict(test_files[i], False)

            # Add the current img detections and scores to the global list
            if len(image_scores) > 0:
                if detections is None:
                    detections = image_detections
                    descriptors = image_desc
                else:
                    detections = np.concatenate((detections, image_detections))
                    descriptors = np.concatenate((descriptors, image_desc))

                scores = np.append(scores, image_scores)
                short_name = ntpath.basename(test_files[i])
                image_names = [short_name for ww in range(len(image_scores))]
                file_names = np.append(file_names, image_names)

            end_time = timeit.default_timer()
            print('Procesing time for image %d/%d is %f sec.'
                % (i, num_test_images, end_time - start_time))               

        return detections, scores, file_names, descriptors

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

    def eval_detections(self, detections, scores, file_names, specialized=False, merge_dir: str|None = None):
        print(f"Evaluating detections...")
        ground_truth_file = np.loadtxt(self.params.path_annotations, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:], int)

        # Only keep gt for faces withing the given range
        if specialized:
            gt_det_bbox_len = np.array([np.sqrt(get_bbox_area(bbox)) for bbox in ground_truth_detections])
            mask = (gt_det_bbox_len >= self.params.dim_window_lower) & (gt_det_bbox_len <= self.params.dim_window_upper)
            ind = np.array([i for i, val in enumerate(mask) if val == 1])
            ground_truth_detections = ground_truth_detections[ind]
            ground_truth_file_names = ground_truth_file_names[ind]
            gt_det_bbox_len = gt_det_bbox_len[ind]


        num_gt_detections = len(ground_truth_detections)
        gt_exists_detection = np.zeros(num_gt_detections)

        # Sort detections by score
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        # Keep only detections on images that have GT annotations(because of specialized)
        valid_mask = np.isin(file_names, ground_truth_file_names)

        detections = detections[valid_mask]
        scores = scores[valid_mask]
        file_names = file_names[valid_mask]

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

        # Set the save directory
        if merge_dir is not None:
            save_dir = merge_dir
        else:
            save_dir = self.params.test_res_dir

        # Save additional information about detections
        results_f = open(os.path.join(save_dir, f"results_{specialized}.txt"), 'w')
        results_f.write(f"Number of gt detections: {len(ground_truth_detections)}, ")
        results_f.write(f"Number of detections: {len(detections)}, ")
        results_f.write(f"True positives: {cum_true_positive[-1]}, ")
        results_f.write(f"False positives: {cum_false_positive[-1]}")
        results_f.flush()

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f' % average_precision)
        plt.savefig(os.path.join(save_dir, f'average_precision_{specialized}.png'))
        plt.close()

    # Returns a string that identifies all the relevant model parameters
    def get_name(self):
        return "_".join(self.params.test_res_dir.split("\\")[-3:])

    def _show_det_on_img(self, detections, scores, img):
        img_cp = img.copy()
        for i in range(len(detections)):
            detection = detections[i]
            cv.rectangle(img_cp, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), thickness=1)
            if scores is not None:
                cv.putText(img_cp, 'score:' + str(scores[i])[:4], (detection[0], detection[1]),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                # cv.putText(img_cp, 'mean color:' + str(np.mean(img[detection[0]:detection[2], detection[1]:detection[3]]))[:4], (detection[2], detection[3]),
                #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
        cv.imshow('image', np.uint8(img_cp))
        cv.waitKey(0)
