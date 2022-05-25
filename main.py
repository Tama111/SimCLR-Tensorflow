import tensorflow as tf

from SimCLR.data_loader import DataLoader
from SimCLR.data_augment import *
from SimCLR.model import SimCLR
from SimCLR.ntxent_loss import NT_Xent_Loss

PATH = 'E://Image Datasets//Celeb A//Dataset//img_align_celeba//img_align_celeba'
dataset = DataLoader(
    DataAugmentPipeline(
        RandomCrop(), 
        LRFlip(), 
        RandomColorJitter(), 
        GrayScale(),
        GaussianBlur2D(11, 5)
), num_images = 100)()

def run():
    resnet50 = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
    simclr = SimCLR(resnet50, 2048, tf.keras.optimizers.Adam(), NT_Xent_Loss())

    losses = simclr.train(t_data, epochs = 3)

    return losses
