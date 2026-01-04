from Parameters import *
from FacialDetector import *
import pdb
from Visualize import *

params: Parameters = Parameters()
params.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
params.dim_hog_cell = 6  # dimensiunea celulei
params.overlap = 0.3

params.threshold = 2.5 # toate ferestrele cu scorul > threshold si maxime locale devin detectii
params.has_annotations = True

params.use_hard_mining = True  # (optional)antrenare cu exemple puternic negative
params.use_flip_images = True  # adauga imaginile cu fete oglindite

# if params.use_flip_images:
#     params.number_positive_examples *= 2

facial_detector: FacialDetector = FacialDetector(params)

pos_desc, neg_desc = facial_detector.get_train_desc()
print(f"Fetched descriptors!")
# CHANGE THIS TO NORMAL!!!
# positive_features = positive_features[:1000]
# negative_features = negative_features[:1000]
# params.number_positive_examples = 1000  # numarul exemplelor pozitive
# params.number_negative_examples = 1000  # numarul exemplelor negative

# Pasul 4. Invatam clasificatorul liniar
training_examples = np.concatenate((np.squeeze(pos_desc), np.squeeze(neg_desc)), axis=0)
train_labels = np.concatenate((np.ones(len(pos_desc)), np.zeros(len(neg_desc))))

# First try without hard mining
params.use_hard_mining = False
facial_detector.train_classifier(training_examples, train_labels)
# detections, scores, file_names = facial_detector.run()
# facial_detector.eval_detections(detections, scores, file_names)
  

# Then with hard mining
# params.use_hard_mining = True
# facial_detector.train_classifier(training_examples, train_labels)
# detections, scores, file_names = facial_detector.run()
# facial_detector.eval_detections(detections, scores, file_names)