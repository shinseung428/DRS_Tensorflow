import tensorflow as tf 
import numpy as np
import cv2
import os
from glob import glob

FLAGS = tf.app.flags.FLAGS

class DataReader():
    def __init__(self):
        if FLAGS.data == 'cifar10':
            path = '/home/sshin/dataset/cifar-10-batches-py'
            images, labels = self.load_cifar10_batch(path)    
        elif FLAGS.data == 'celeb':
            path = '/home/sshin/dataset/img_align_celeba'
            images = self.load_celeb(path)
            labels = None
        else:
            print ("Invalid data!!")

        self.images = images
        self.labels = labels


    def prepare_data(self):
        print ("Preparing Dataset...")
        images = []
        for img in self.images:
            img = img / 127.5 - 1
            images.append(cv2.resize(img, (FLAGS.img_hw, FLAGS.img_hw)))
        return images 

    def load_celeb(self, path):
        img_paths = glob(os.path.join(path, '*.jpg'))
        images = []
        for path in img_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = img[16:-16,16:-16,:]
            images.append(image)
        return images
    
    def load_cifar10_batch(self, path):
        import pickle
        all_batch = []
        all_labels = []
        for batch_id in range(1,6):
            with open(path + '/data_batch_' + str(batch_id), mode='rb') as file:
                # note the encoding type is 'latin1'
                batch = pickle.load(file, encoding='latin1')
            
            features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
            labels = batch['labels']
            all_batch.append(features)
            all_labels.append(labels)

        img_batch = np.concatenate(all_batch, axis=0)
        label_batch = np.concatenate(all_labels, axis=0)
        
        return img_batch, label_batch
