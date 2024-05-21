import os
parent = os.getcwd()

# GENERATE IMAGES
type_generate = "random_chars" # Options: ["random_chars", "random_words"]
multiclass = True # Options: [True, False]
folder = "oneclass_randomchars"
n_train_images = 2
n_test_images = 1

train_labels = f"{parent}/Datasets/{folder}/labels/train"
train_images = f"{parent}/Datasets/{folder}/images/train"
test_labels = f"{parent}/Datasets/{folder}/labels/test"
test_images = f"{parent}/Datasets/{folder}/images/test"
path_fonts = f"{parent}/generate_images/fonts/"

# DETECTION (OCR)
ocr_predictions = f"{parent}/Datasets/ocr_predictions"

# DETECTION (YOLO)
data_yaml_detection = f"{parent}/Detection/YOLO/data.yaml"
path_yolo_detection = f"{parent}/Datasets/{folder}/"

# RECOGNITION (CNN)
saved_model_cnn = f"{parent}Recognition/saved_model"

# PIPELINE (OCR+CNN i YOLO+CNN)
dataset = "iiit" # Options: ["iiit", "ours"]
cnn_model = "resnet_iit"
model_cnn_entrenat = f'{parent}/Recognition/saved_model/{cnn_model}.pt'
yolo_model = "best_iit"
model_yolo_entrenat = f"{parent}/Detection/YOLO/trained_models/{yolo_model}.pt"
ocr_cnn_store_files = f"{parent}/PIPELINE/OCR_CNN"
yolo_cnn_store_files = f"{parent}/PIPELINE/YOLO_CNN"

# PIPELINE (YOLO FOR RECOGNITION AND DETECTION)
data_yaml_pipeline = f"{parent}/PIPELINE/YOLO/data.yaml"
path_yolo_pipeline = f"{parent}/Datasets/{folder}/"
yolo_pipeline_model = "char_best"
yolo_entrenat_recog_detect = f"{parent}/PIPELINE/YOLO/trained_models/{yolo_pipeline_model}.pt"
yolo_pipeline_store_files = f"{parent}/PIPELINE/YOLO"

# PIPELINE (PHOCNET)
saved_model_phocnet = f"{parent}/PIPELINE/PHOCNET/trained_models/"
store_knn_classifier = f"{parent}/PIPELINE/PHOCNET/utils/knn_classifier"
lexicon_file = f"{parent}/Datasets/lexicon.txt"
bigrams_file = f"{parent}/PIPELINE/PHOCNET/utils/bigrams_file.txt"

# PIPELINE (CRNN)
type_crnn_dataset = "ours" # Options: ["iiit", "ours"]
mat_data_train_file = ""
mat_data_test_file = ""
mat_data_img_dir = ""