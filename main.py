import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_addons as tfa
import glob, random, os, warnings
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils import vision_transformer
from tqdm import tqdm
import openslide
from openslide import OpenSlide
from sklearn.model_selection import train_test_split

def seed_everything(seed = 0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything()
warnings.filterwarnings('ignore')

learning_rate = 0.0001
weight_decay = 0.0001
num_epochs = 50

image_size = 224
batch_size = 12
n_classes = 1

df_train = pd.read_csv('../input/mayo-clinic-strip-ai/train.csv')
df_val  = pd.read_csv('../input/mayo-clinic-strip-ai/test.csv')

df_train["file_path"] = df_train["image_id"].apply(lambda x: "../input/mayo-clinic-strip-ai/train/" + x + ".tif")
df_val["file_path"]  = df_val["image_id"].apply(lambda x: "../input/mayo-clinic-strip-ai/test/" + x + ".tif")

df_train["target"] = df_train["label"].apply(lambda x : 1 if x=="CE" else 0)

%%time
def preprocess(image_path):
    slide=OpenSlide(image_path)
    region= (1000,1000)    
    size  = (5000, 5000)
    image = slide.read_region(region, 0, size)
    image = tf.image.resize(image, (image_size, image_size))
    image = np.array(image)    
    return image

x_train=[]
for i in tqdm(df_train['file_path']):
    x1=preprocess(i)
    x_train.append(x1)

x_train=np.array(x_train)
y_train=df_train['target']

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.1)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
          
                zoom_range=0.3,


#                                                           preprocessing_function = data_augment
)

train_gen = train_datagen.flow_from_dataframe(dataframe = df_train,
                                        x_col='Ids',
                                        y_col='label',
#                                         subset = 'training',
                                        batch_size = batch_size,
                                        seed = 1,
                                        color_mode = 'rgb',
                                        shuffle = True,
                                        class_mode = 'binary',
                                        target_size = (image_size, image_size))

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#                 rotation_range=30,
#                 width_shift_range=0.5,
#                 height_shift_range=0.5,
#                 zoom_range=0.3,
#                 samplewise_center = True,
#                 samplewise_std_normalization = True,
#                                                           preprocessing_function = data_augment
)

valid_gen = valid_datagen.flow_from_dataframe(dataframe = df_val,
                                        x_col='Ids',
                                        y_col='label',
#                                         subset = 'validation',
                                        batch_size = batch_size,
                                        seed = 1,
                                        color_mode = 'rgb',
                                        shuffle = False,
                                        class_mode = 'binary',
                                        target_size = (image_size, image_size))


STEP_SIZE_TRAIN = train_gen.n// train_gen.batch_size
STEP_SIZE_VALID = valid_gen.n// valid_gen.batch_size

decay_steps = train_gen.n // train_gen.batch_size/ 

lr_decayed_fn = tf.keras.experimental.CosineDecay(learning_rate, decay_steps)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_decayed_fn)

earlystopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                 min_delta = 1e-4,
                                                 patience = 5,
                                                 restore_best_weights = True,
                                                 verbose = 1)

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = './model1.hdf5',
                                                  monitor = 'val_loss', 
                                                  verbose = 1, 
                                                  save_best_only = True,
                                                  save_weights_only = True,
                                                 )

reduce_lr=ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=2, min_delta=1e-4, verbose=1, min_lr=1e-7)

callbacks = [earlystopping, lr_scheduler, checkpointer, reduce_lr]

model = vision_transformer()
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics='binary_accuracy'
)

model.fit(
          x = train_gen,
          steps_per_epoch = STEP_SIZE_TRAIN,
          validation_data = valid_gen,
          validation_steps = STEP_SIZE_VALID,
          epochs = num_epochs,
          callbacks = callbacks
         )

