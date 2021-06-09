import torch
import os
import numpy as np
from fastai.vision import *
from fastai.metrics import error_rate, accuracy
from tqdm import tqdm
import cv2
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.models as models

# model=torch.load('vgg16_bn.pth')
dataset_path = 'dataset_with_mask'

trfm = get_transforms(do_flip=True, flip_vert=True,
                      max_zoom=1.2, max_rotate=20.0, max_lighting=0.4)

data = ImageDataBunch.from_folder(
    dataset_path, train='.', valid_pct=0.2, bs=4).normalize(imagenet_stats)
print(data.classes)
learn = cnn_learner(data, models.vgg16_bn, metrics=[error_rate])

learn.fit_one_cycle(4)

learn.save('v1_be')
learn.export()
