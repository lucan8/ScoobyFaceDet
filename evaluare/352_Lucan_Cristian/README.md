# Scooby Doo detector

## Dependencies

These are the dependencies (if something is missing just run pip install ultralytics as I don't know what other packages it installed)

torch==2.9.1
torchvision==0.24.1
ultralytics==8.4.5
ultralytics-thop==2.0.18
scikit-image==0.26.0
scikit-learn==1.8.0
numpy==2.2.6
opencv-python==4.12.0.88
matplotlib==3.10.6
joblib==1.5.3

## Project strcuture

### Tasks

Both task1 and task2 have the same structure:

base_dir: 352_Lucan_Cristian  

task_dir: base_dir/task_i  

script: base_dir/task_i/RunProject.py  

function: run_project() - expects "antrenare", "validare" and "testare" to be at ../../../
(so if my project is in root_dir/evaluare/352_Lucan_Cristian/task_i, the data should be in root_dir)  

output: everything is saved to base_dir/saved_files, but here the tasks differ  

task1: The results of a given model are found in {descriptors_param_dir}/{model_params_dir}/{test_res_dir}/data
example:  

evaluare\352_Lucan_Cristian\saved_files\36_6_2_0_0.25_1.0_64_18_2_True_False_False\model_4_10_2_0.7_1e-05_True\test_res_0_1.2_0.9\data  

Note: The detections of the best models are actually in merge_results/{hash}/data
example:  

evaluare\352_Lucan_Cristian\saved_files\merge_results\a3e5101fe0249c3a3d36dbe18c50dcac\data  

task2: The detections of the models are actually in merge_results_classifier/{hash}/data/{char_name}  

example:  

evaluare\352_Lucan_Cristian\saved_files\merge_results_classifier\d7fdc02165db17a78300c27351b92/daphne/detections.npy

Note: For task1 it is advised that you use the cache, so set use_cache=True to save some time though be prepared to give away quite a bit of storage(at least 20GB)!  

In their current states, both tasks will just load detections from the dirs so if you want to retrain or re-run the detector just remove the test results dir or the whole model dir.  

### Bonus

script: bonus.py - Trains and runs predictions using yolo, for both task1 and task2 at the same time  

function: run_project()
output: The output of both tasks is bonus_dir=saved_files/bonus/results/data, but it again differs slightly based on task,  

for task1 you will find them directly in the specified dir, for task2 in {bonus_dir}/{char_name}  

path example for  

task1:evaluare\352_Lucan_Cristian\saved_files\bonus\results\data\detections_all_faces.npy  

task2:evaluare\352_Lucan_Cristian\saved_files\bonus\results\data\daphne\detections.npy  
