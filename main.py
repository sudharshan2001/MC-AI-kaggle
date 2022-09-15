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
