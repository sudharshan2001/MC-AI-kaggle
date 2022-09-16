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
