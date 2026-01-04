import os

class Parameters:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.train_dir = os.path.join(self.base_dir, "antrenare")
        self.test_dir = os.path.join(self.base_dir, 'testare')
        self.val_dir = os.path.join(self.base_dir, 'validare')
        self.dir_save_files = os.path.join(self.base_dir, 'saved_files')
        self.pos_desc_path = os.path.join(self.dir_save_files, "pos_desc")
        self.all_pos_desc_file = os.path.join(self.pos_desc_path, "all.npy")
        self.all_neg_desc_file = os.path.join(self.dir_save_files, "neg_desc.npy")

        if not os.path.exists(self.pos_desc_path):
            os.mkdir(self.pos_desc_path)
            print("Created directory for saved files!")

        # set the parameters
        self.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_hog_cell = 8  # dimensiunea celulei
        self.dim_descriptor_cell = 64  # dimensiunea descriptorului unei celule
        self.dim_block = 2
        self.neg_patch_count_per_img = 15
        self.neg_rand_perc = 0.7 # How many of the patches will be random, the rest will be near face
        self.overlap = 0.3
        self.has_annotations = False
        self.threshold = 0
        self.use_hard_mining = False  
        self.use_flip_images = True 
