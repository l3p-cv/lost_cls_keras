import tensorflow as tf
import os
import lost_ds as lds
import json

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D

from config import *


# model class names to json
def merch_save_classes(class_dict, ds, path):
    for label in list(ds.unique_labels(col='img_lbl')):
        if not label in class_dict.keys():
            new_id = len(class_dict)
            class_dict[label] = new_id
        
    with open(path, 'w') as fp:
        json.dump(class_dict, fp)
 
# load anno data
anno_data = []

for root, _, files in os.walk(os.path.abspath(TRAIN_ANNO_DATA_PATH)):
    for file in files:
        if file.endswith(('.parquet')):
            anno_data.append(os.path.join(root, file))
        
init_ds = lds.LOSTDataset(anno_data)

ds = lds.LOSTDataset(lds.remap_img_path(init_ds.df, 
                                        new_root_path=TRAIN_IMG_PATH, 
                                        col='img_path'))

# build class dictionary
if not TRAIN_FROM_CHECKPOINT:
    class_dict = {}
    
    merch_save_classes(class_dict, ds, TRAIN_LABEL_MAP)
    
else:
    with open(TRAIN_LABEL_MAP, 'r') as fp:
        class_dict = json.load(fp)
        
    merch_save_classes(class_dict, ds, TRAIN_LABEL_MAP)   

with open(TRAIN_LABEL_MAP, 'r') as fp:
        class_dict = json.load(fp)

# remap label 
ds = lds.LOSTDataset(lds.remap_labels(ds.df, class_dict,  col='img_lbl', dst_col='img_lbl_mapped'))

# data preprocess
train_datagen = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=30,
        zoom_range=0.3,
        width_shift_range=(-0.05,0.05),
        height_shift_range=(-0.05,0.05),
        shear_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.20,
        brightness_range=(0.8, 1.0)
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=ds.df,
    directory=None,
    x_col="img_path",
    y_col="img_lbl_mapped",
    target_size=(TRAIN_INPUT_SIZE, TRAIN_INPUT_SIZE),
    batch_size=TRAIN_BATCH_SIZE,
    class_mode="categorical",
    subset='training',
    shuffle=True,
    seed=42
)

valid_generator = train_datagen.flow_from_dataframe(
    dataframe=ds.df,
    directory=None,
    x_col="img_path",
    y_col="img_lbl_mapped",
    target_size=(TRAIN_INPUT_SIZE, TRAIN_INPUT_SIZE),
    batch_size=TRAIN_BATCH_SIZE,
    class_mode="categorical",
    subset='validation',
    shuffle=True,
    seed=42
)

# callback for best model
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME),
    monitor='val_categorical_accuracy',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    save_freq='epoch',
    options=None,
    initial_value_threshold=None
)

# check model version in repo
versions = []
version_path = os.path.join(MODEL_PATH, TRAIN_MODEL_NAME)
for _, dirs, _ in os.walk(version_path):
    for dir in dirs:
        try:
            versions.append(int(dir))
        except:
            continue

versions.sort()

if not versions:
    highest_version = 1
else:
    highest_version = versions[-1] + 1
    

# train initial
if not TRAIN_FROM_CHECKPOINT:
    
    # build model
    base_model = tf.keras.applications.MobileNet(input_shape=(TRAIN_INPUT_SIZE, TRAIN_INPUT_SIZE, 3),
                                                include_top=False,
                                                weights="imagenet",
                                                classes=TRAIN_CLASS_NUMBERS,
                                                classifier_activation="softmax")



    inputs = base_model.input

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    outputs = Dense(TRAIN_CLASS_NUMBERS, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    model.compile(optimizer=keras.optimizers.Adam(),
                loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=[keras.metrics.CategoricalAccuracy()])

    # train
    model.fit(train_generator, epochs=TRAIN_EPOCHS, callbacks=model_checkpoint_callback, validation_data=valid_generator)

# train from checkpoint
else:
    
    # load saved model
    model_path=os.path.join(MODEL_PATH, TRAIN_MODEL_NAME, str(highest_version - 1), 'model.savedmodel')
    model = keras.models.load_model(model_path)

    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=[keras.metrics.CategoricalAccuracy()])

    # train
    model.fit(train_generator, epochs=TRAIN_EPOCHS, callbacks=model_checkpoint_callback, validation_data=valid_generator)

# save best model
model_path=os.path.join(MODEL_PATH, TRAIN_MODEL_NAME, str(highest_version), 'model.savedmodel')

model.load_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))

model.save(model_path)