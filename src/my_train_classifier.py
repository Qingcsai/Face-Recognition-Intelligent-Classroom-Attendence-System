# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 20:28:00 2019
实现简易版人脸注册，即训练分类器，参考classifier.py,最终得到.pkl模型
@author: ASUS
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC


seed = 666
data_dir = 'F:/Face_Recognition/FaceNet/facenet-master/data/behind_160'
#model = 'F:/Face_Recognition/FaceNet/facenet-master/src/20180402-114759'
classifier_filename = 'F:/Face_Recognition\FaceNet/facenet-master/pre-trained-model/behind.pkl'
#image_size = 160
#batch_size = 1000

def train(data_dir, classifier_filename):
    model = 'F:/Face_Recognition/FaceNet/facenet-master/src/20180402-114759'
    image_size = 160
    batch_size = 1000
    with tf.Graph().as_default():
        with tf.Session() as sess:
#                np.random.seed(seed = 666)
            
#            if args.use_split_dataset:
#                dataset_tmp = facenet.get_dataset(args.data_dir)
#                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class, args.nrof_train_images_per_class)
#                if (args.mode=='TRAIN'):
#                    dataset = train_set
#                elif (args.mode=='CLASSIFY'):
#                    dataset = test_set
#            else:
            dataset = facenet.get_dataset(data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset'
                paths, labels = facenet.get_image_paths_and_labels(dataset)
            
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            classifier_filename_exp = os.path.expanduser(classifier_filename)

#            if (args.mode=='TRAIN'):
                # Train classifier
            print('Training classifier')
            model = SVC(kernel='linear', probability=True)
            model.fit(emb_array, labels)
            
                # Create a list of class names
            class_names = [ cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)

if __name__ == '__main__':
    train(data_dir,classifier_filename)

            