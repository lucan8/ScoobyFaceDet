import os
import numpy as np
class Parameters:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.train_dir = os.path.join(self.base_dir, "antrenare")
        self.test_dir = os.path.join(self.base_dir, 'testare')
        self.val_dir = os.path.join(self.base_dir, 'validare')
        self.dir_save_files = os.path.join(self.base_dir, 'saved_files')
        self.bbox_len_list_file = os.path.join(self.dir_save_files, "bbox_len_list.npy")
        self.path_annotations = os.path.join(self.val_dir, "task1_gt_validare.txt")
        
        # Hog/window parameters
        self.dim_hog_cell = 6
        self.dim_descriptor_cell = 6
        self.dim_block = 2
        self.block_stride = 1
        self.use_flip_images = True
        self.use_clahe = False
        self.use_lbp = False

        # Descriptos parameters
        self.soft_neg_overlap = 0.15 # How much overlap is acceptable for a negative face example
        self.neg_patch_count_per_img = 0
        self.soft_neg_patch_perc = 1.0
        self.hard_neg_img_count_per_char = 100
        self.hard_count_per_img = 10
        self.hard_pos_overlap = 0.7
        self.neg_patch_factor = 3
        self.use_cache = True
        # self.same_neg_thr = True # Use the same threshold for both hard and soft negatives
        
        # Run parameters(train + predict/test params)
        self.overlap = 0.3
        self.has_annotations = False
        self.threshold = 0
        self.hard_neg_mining_it_count = 0
        self.image_resizes = [] # Set this in run
        self.use_all_hard_min_resizes = True
        self.window_count = 8
        self.down_step = 0.9
        self.up_step = 1.3
        self.hard_pos_overlap_base = 0.7
        self.hard_pos_overlap_step = 0.15 # Grow threshold with each iteration
        self.dim_window = 64 # It only marks the max length of the windows that are taken into account
        self.class_weights = None
        self.SVC_iter = 1000
        self.tolerance_lower = 0
        self.tolerance_upper = 0
        self.two_faced = True
        

    def _set_hard_min_data_dir(self):
        self.hard_min_data_dir = os.path.join(self.model_dir, "hard_min_data")
        self.hard_min_pos_desc_file = os.path.join(self.hard_min_data_dir, "pos_desc.npy")
        self.hard_min_neg_desc_file = os.path.join(self.hard_min_data_dir, "neg_desc.npy")
        
        if not os.path.exists(self.hard_min_data_dir):
            os.makedirs(self.hard_min_data_dir, True)
            print("Created directories for hard mined data!")

    def _set_face_class_dir(self):
        self.classifier_dir = os.path.join(self.test_res_dir_all, "classifier")

        # Model dirs
        self.class_model_dir = os.path.join(self.classifier_dir, f"model_0_{self.class_weights}_{self.SVC_iter}_{self.tolerance_lower}_{self.tolerance_upper}_{self.two_faced}")
        self.class_model_file = os.path.join(self.class_model_dir, "model")
        if not os.path.exists(self.class_model_dir):
            os.makedirs(self.class_model_dir)
            print("Created dirs for classifier model!")

        # Test results dirs
        self.test_res_data_dir_split = os.path.join(self.class_model_dir, "data")
        
        if not os.path.exists(self.test_res_data_dir_split):
            os.makedirs(self.test_res_data_dir_split)
            print("Created dirs for classifier test results")

    def reset_model_dir(self, hard_min_it: int):
        # Test and model dir/files
        if hard_min_it == 0:
            self.model_dir = os.path.join(self.run_dir, f"model_0")
        else:
            self.model_dir = os.path.join(self.run_dir, f"model_{hard_min_it}_{self.hard_count_per_img}_{self.hard_neg_img_count_per_char}_{self.hard_pos_overlap_base}_{self.hard_pos_overlap_step}_{self.use_all_hard_min_resizes}")
        self.model_file = os.path.join(self.model_dir, "model")

        # Set directories for face/no-face detections
        self.test_res_dir_all = os.path.join(self.model_dir, f"test_res_{self.threshold}_{self.up_step}_{self.down_step}")
        self.test_res_data_dir_all = os.path.join(self.test_res_dir_all, "data")
        
        # Set directories for face classifier
        self._set_face_class_dir()
        
        # Hard mining data dirs/files
        if hard_min_it > 0:
            self._set_hard_min_data_dir()

    def set_window_stuff(self, dim_window: int, dim_window_upper: int, down_step: float, up_step: float=0, dim_window_lower: float=0):
        self.dim_window = dim_window
        self.dim_window_upper = dim_window_upper
        self.down_step = down_step
        self.up_step = up_step

        if dim_window_lower:
            self.dim_window_lower = dim_window_lower
        else:
            self.dim_window_lower = dim_window

    def set_cache_dir(self):
        self.cache_dir = os.path.join(self.dir_save_files, "cache", f"{self.dim_hog_cell}_{self.dim_block}")
        self.train_cache = os.path.join(self.cache_dir, "train")
        self.val_cache = os.path.join(self.cache_dir, "validation")

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.train_cache)
            os.makedirs(self.val_cache)
            print("Created directories for train and val feature caching!")

    # This should be called everytime you change a parameter
    def set_run_dirs(self):
        self.run_dir = os.path.join(self.dir_save_files, f"{self.dim_window}_{self.dim_hog_cell}_{self.dim_block}_{self.neg_patch_count_per_img}_{self.soft_neg_overlap}_{self.soft_neg_patch_perc}_{self.dim_window_upper}_{self.dim_window_lower}_{self.neg_patch_factor}_{self.use_flip_images}_{self.use_clahe}_{self.use_lbp}")
        self.pos_desc_dir = os.path.join(self.run_dir, "pos_desc")
        self.all_pos_desc_file = os.path.join(self.pos_desc_dir, "all.npy")
        self.all_neg_desc_file = os.path.join(self.run_dir, "neg_desc.npy")
        
        self.set_cache_dir()
        self.reset_model_dir(self.hard_neg_mining_it_count)

        if not os.path.exists(self.pos_desc_dir):
            os.makedirs(self.pos_desc_dir)
            print("Created directories for saved descriptors!")