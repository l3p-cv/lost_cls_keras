# Train options
TRAIN_FROM_CHECKPOINT           = False
TRAIN_MODEL_NAME                = "classification_model"
TRAIN_IMG_PATH                  = "/home/dgacon/Dokumente/02_Projekt/00_projekt_daten/00_Images/marvel_ds"
TRAIN_ANNO_DATA_PATH            = "anno_data/"
TRAIN_LABEL_MAP                 = "anno_data/lbl_map.json"
TRAIN_INPUT_SIZE                = 224 # tensorflow model supports 128, 160, 192, 224
TRAIN_BATCH_SIZE                = 16
TRAIN_CHECKPOINTS_FOLDER        = "checkpoints"
TRAIN_EPOCHS                    = 100

#  Model options
MODEL_PATH                      = "/home/dgacon/server/docs/model_repo/detection/marvel_2" 
