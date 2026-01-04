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
from skimage.feature import hog

# TODO: Add the possibility to re-do only the negative descriptors
# TODO: Hard mining
# TODO: Statistical analysis over bounding boxes
# Transform the dictionary into a list
def merge_dict(dic: dict[str, np.ndarray]):
    return np.concatenate([v for k, v in dic.items()])

# Saves the value at save_path/dic_key
def save_dictionary(save_path: str, dic: dict[str, np.ndarray]):
    for file_name in dic:
        np.save(os.path.join(save_path, file_name), dic[file_name])


class FacialDetector:
    detection_size = 4
    detection_area_list = [] # Keep track of all detection bbox areas for statistical purposes

    def __init__(self, params:Parameters):
        self.params = params
        self.best_model = None
        self.image_resizes = [0.3, 0.5, 0.7, 1, 1.2, 1.5, 2]

    # Returns the positive and negative descriptors for the training data as 2 list
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
                pos_desc, neg_desc = self._get_pos_desc(img, detections, char_names), self._get_neg_desc(img, detections)
                
                # Add them to bigger dict/list
                self._add_pos_desc(all_pos_desc, pos_desc)
                all_neg_desc.extend(neg_desc)

                last_ind = i
        
        return all_pos_desc, np.array(all_neg_desc)

    # Returns the two lists containing the positive and negative descriptors
    def _fetch_desc(self):
        return np.load(self.params.all_pos_desc_file), np.load(self.params.all_neg_desc_file)

    # Returns a list of hog descriptors for random negative patches
    # TODO: Add support for near-face patches as well 
    def _get_neg_desc(self, img: np.ndarray, bboxes: np.ndarray):
        H, W = img.shape
        
        sample_count = int(self.params.neg_patch_count_per_img * self.params.neg_rand_perc / len(bboxes))
        max_tries_per_sample = 100
        iou_thr = 0.05

        neg_patches = []
        for _ in range(sample_count):
            tries = 0

            while tries <= max_tries_per_sample:
                tries += 1

                x = np.random.randint(0, W - self.params.dim_window)
                y = np.random.randint(0, H - self.params.dim_window)
                cand = [x, y, x + self.params.dim_window, y + self.params.dim_window]

                # reject if overlaps any face
                if all(self.intersection_over_union(cand, bb) < iou_thr for bb in bboxes):
                    patch = img[cand[1]:cand[3], cand[0]:cand[2]]
                    patch = hog(patch, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                cells_per_block=(self.params.dim_block, self.params.dim_block), feature_vector=True)
                    neg_patches.append(patch)
                    break

        return np.array(neg_patches)

    # Returns the positive descriptors for the detections in img as a dictionary
    # key: char_name, value: pos_desc
    def _get_pos_desc(self, img: np.ndarray, detections: np.ndarray, char_names: np.ndarray):
        pos_desc_split = {name:[] for name in char_names}

        for i in range(len(detections)):
            bbox, char_name = detections[i], char_names[i]

            # This will help compute the window dimensions
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            FacialDetector.detection_area_list.append(h * w)

            # Extract face and resize it to window dimensions
            face = cv.resize(img[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy(), (self.params.dim_window, self.params.dim_window), interpolation=cv.INTER_AREA)
            # Extract hog desc
            face_features = hog(face, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                cells_per_block=(self.params.dim_block, self.params.dim_block), feature_vector=True)
            # Add it to the dictionary
            pos_desc_split[char_name].append(face_features)

            # The same for flipped face
            if self.params.use_flip_images:
                face_features = hog(np.fliplr(face), pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                    cells_per_block=(self.params.dim_block, self.params.dim_block), feature_vector=True)
                pos_desc_split[char_name].append(face_features)
        

        pos_desc_split = {name:np.array(pos_desc_split[name]) for name in pos_desc_split}
        return pos_desc_split

    # Adds the positive desc in pos_desc to all_pos_desc, changing it
    def _add_pos_desc(self, all_pos_desc:dict[str, np.ndarray], pos_desc: dict[str, np.ndarray]):
        for char_name in pos_desc:
            if char_name in all_pos_desc:
                all_pos_desc[char_name] = np.concatenate((all_pos_desc[char_name], pos_desc[char_name]))
            else:
                all_pos_desc[char_name] = pos_desc[char_name]

    # TODO: CHANGE
    # Run the detector on the negative examples and return the hog for each detection
    def get_hard_neg_desc(self):
        images_path = os.path.join(self.params.train_dir, '*.jpg')
        files = glob.glob(images_path)
        
        num_images = len(files)
        max_false_pos = 3

        neg_desc = []
        
        # Make a softer threshold to get more false predictions
        init_thr = self.params.threshold
        self.params.threshold = 0
        
        # Read each negative image, get predictions and add the descriptors to neg_desc 
        for i in range(num_images):
            
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            _, scores, hog_desc = self.predict(img, False)

            print(f"Hard mined image {i}")
            # Skip if no detection
            if len(scores) == 0:
                continue
            
            # Add the top max_false_pos descriptors
            sorted_score_ind = np.argsort(scores)[::-1]
            neg_desc.extend(hog_desc[sorted_score_ind][:max_false_pos])

        # Reset threshold
        self.params.threshold = init_thr
        
        return neg_desc

    # Returns the linear SVC with the most accuracy, each svc differs by the C param
    def _train(self, training_examples, train_labels):
        best_accuracy = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2, 10 ** -1, 10 ** 0]
        for c in Cs:
            model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)

            acc = model.score(training_examples, train_labels)
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = deepcopy(model)

        return best_model, best_accuracy
        
    # Trains and saves a linear classifier
    def train_classifier(self, training_examples, train_labels):
        if os.path.exists(self.params.model_file):
            self.best_model = pickle.load(open(self.params.model_file, 'rb'))
            return

        print(f"Training linear classifier...")

        # Train a linear SVC
        self.best_model, best_acc = self._train(training_examples, train_labels)
        print(f"Model accuracy: {best_acc}")
        self.plot_train(training_examples, train_labels)

        # Hard negative mining
        if self.params.use_hard_mining:
            it_count = 3
            print(f"Hard negative mining for {it_count} iterations...\n")
            
            for i in range(it_count):
                # Get descriptors for hard negatives
                start_time = timeit.default_timer()
                desc = self.get_hard_neg_desc()
                end_time = timeit.default_timer()
                print(f"Duration: {end_time - start_time} sec")

                # Add the new negative samples, labels and retrain
                if len(desc):
                    desc = np.array(desc).reshape(len(desc), -1)
                    training_examples = np.vstack((training_examples, desc))
                    train_labels = np.concatenate((train_labels, np.zeros(len(desc))))
                    self.best_model, _ = self._train(training_examples, train_labels)

        # salveaza clasificatorul
        pickle.dump(self.best_model, open(self.params.model_file, 'wb'))
            
    # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
    # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
    def plot_train(self, training_examples, train_labels):
        scores = self.best_model.decision_function(training_examples)
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]

        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(positive_scores)))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()

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
        return sorted_image_detections[is_maximal], sorted_scores[is_maximal], image_hog_desc[is_maximal]

    # Returns the bounding boxes, scores and hog descriptors of the found faces, each as a list
    def predict(self, img_init, use_nms = True):
        image_scores = []
        image_detections = []
        image_desc = []

        w = self.best_model.coef_.T
        bias = self.best_model.intercept_[0]

        hog_offset = self.params.dim_hog_cell // 2

        # Resize image to simulate using different window sizes
        for resize in self.image_resizes:
            print(f"For image resize {resize}")
            img = cv.resize(img_init, None, fx=resize, fy=resize, interpolation=cv.INTER_LINEAR)

            hog_descriptors = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                cells_per_block=(self.params.dim_block, self.params.dim_block), feature_vector=False)
            
            num_rows, num_cols = hog_descriptors.shape[:2]
            num_cell_in_template = self.params.dim_window // self.params.dim_hog_cell - self.params.dim_block + 1
            
            # Slide window over hogged resized image
            for y in range(0, num_rows - num_cell_in_template):
                for x in range(0, num_cols - num_cell_in_template):
                    descr = hog_descriptors[y:y + num_cell_in_template, x:x + num_cell_in_template].flatten()
                    score = np.dot(descr, w)[0] + bias

                    # Get bounding box for positive detections
                    if score > self.params.threshold:
                        x_min = int((x * self.params.dim_hog_cell - hog_offset) / resize)
                        y_min = int((y * self.params.dim_hog_cell - hog_offset) / resize)
                        x_max = int((x * self.params.dim_hog_cell + self.params.dim_window - hog_offset) / resize)
                        y_max = int((y * self.params.dim_hog_cell + self.params.dim_window - hog_offset) / resize)

                        # Might go out of bounds with the window
                        if x_min < img_init.shape[1] and y_min < img_init.shape[0]:
                            image_detections.append([x_min, y_min, x_max, y_max])
                            image_scores.append(score)
                            image_desc.append(descr)

        # Transorm to numpy arrays
        image_detections, image_scores, image_desc = np.array(image_detections), np.array(image_scores), np.array(image_desc)
        # Run non maximal suppression
        if use_nms and len(image_scores) > 0:
            image_detections, image_scores, image_desc = self.non_maximal_suppression(
                                                                          image_detections,
                                                                          image_scores,
                                                                          image_desc,
                                                                          img_init.shape)
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
        
        test_files = glob.glob(test_images_path)

        detections = None
        scores = np.array([])
        file_names = np.array([])

        num_test_images = len(test_files)
        
        for i in range(num_test_images):
            start_time = timeit.default_timer()

            print('Processing image %d/%d..' % (i, num_test_images))
            img = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)
            
            image_detections, image_scores, _ = self.predict(img)

            # Add the current img detections and scores to the global list
            if len(image_scores) > 0:
                if detections is None:
                    detections = image_detections
                else:
                    detections = np.concatenate((detections, image_detections))
                scores = np.append(scores, image_scores)
                short_name = ntpath.basename(test_files[i])
                image_names = [short_name for ww in range(len(image_scores))]
                file_names = np.append(file_names, image_names)

                
                # for detection in image_detections:
                #     cv.rectangle(img_init, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), thickness=1)
                
                # cv.imshow('image', np.uint8(img_init))
                # cv.waitKey(0)    

            end_time = timeit.default_timer()
            print('Procesing time for image %d/%d is %f sec.'
                % (i, num_test_images, end_time - start_time))               

        return detections, scores, file_names

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

    def eval_detections(self, detections, scores, file_names):
        print(f"Evaluating detections...")
        ground_truth_file = np.loadtxt(self.params.path_annotations, dtype='str')
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

        print(f"Number of detections: {len(detections)}")
        print(f"True positives: {cum_true_positive[-1]}")
        print(f"False positives: {cum_false_positive[-1]}")

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f' % average_precision)
        plt.savefig(os.path.join(self.params.run_dir, 'precizie_medie.png'))
        plt.show()
