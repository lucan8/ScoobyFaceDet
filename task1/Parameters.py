import os

class Parameters:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.train_dir = os.path.join(self.base_dir, "antrenare")
        self.test_dir = os.path.join(self.base_dir, 'testare')
        self.val_dir = os.path.join(self.base_dir, 'validare')
        self.dir_save_files = os.path.join(self.base_dir, 'saved_files')
        self.path_annotations = os.path.join(self.val_dir, "task1_gt_validare.txt")

        # set the parameters
        self.dim_window = 64  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_hog_cell = 8  # dimensiunea celulei
        self.dim_descriptor_cell = 8  # dimensiunea descriptorului unei celule
        self.dim_block = 2
        self.neg_patch_count_per_img = 7 # The actual one is this / nr_faces_in_img
        self.neg_rand_perc = 0.7 # How many of the patches will be random, the rest will be near face
        self.overlap = 0.3
        self.has_annotations = False
        self.threshold = 0
        self.use_hard_mining = False  
        self.use_flip_images = True
        self.set_run_dirs()
    
    # This should be called everytime you change a parameter that is relevant 
    # for feature extraction or training
    # Set and create the directories neccessary for the current run
    # These hold the positive and negative descriptors and the trained model
    def set_run_dirs(self):
        self.run_dir = os.path.join(self.dir_save_files, f"{self.dim_window}.{self.dim_hog_cell}.{self.dim_block}.{self.neg_patch_count_per_img}.{self.neg_rand_perc}.{self.use_flip_images}.{self.use_hard_mining}")
        self.pos_desc_dir = os.path.join(self.run_dir, "pos_desc")
        self.all_pos_desc_file = os.path.join(self.pos_desc_dir, "all.npy")
        self.all_neg_desc_file = os.path.join(self.run_dir, "neg_desc.npy")
        self.model_file = os.path.join(self.run_dir, "model")

        if not os.path.exists(self.pos_desc_dir):
            os.makedirs(self.pos_desc_dir)
            print("Created directories for run files!")

        

