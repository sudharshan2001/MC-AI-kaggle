import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_addons as tfa
import glob, random, os, warnings
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tqdm import tqdm
import openslide
from openslide import OpenSlide
import glob
import os
from sklearn.model_selection import train_test_split

def seed_everything(seed = 0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything()
warnings.filterwarnings('ignore')


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
                                        class_mode = 'categorical',
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
                                        class_mode = 'categorical',
                                        target_size = (image_size, image_size))



decay_steps = train_gen.n // train_gen.batch_size/ 

initial_learning_rate = learning_rate

lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate, decay_steps)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_decayed_fn)
