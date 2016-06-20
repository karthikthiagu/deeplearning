# Import system libraries
import os
import sys
import gzip
import cPickle as pkl

# Import external libraries
import numpy as np
import cv2

data = pkl.load(gzip.open('./data/mnist.pkl.gz', 'r'))

# Explore the data-type
print 'type of data : %s' % type(data)
print 'length of data = %s' % len(data)
print 'length of train = %s' % str(len(data[0]))
print 'size of train examples = %s, size of labels = %s' % (str(data[0][0].shape), str(data[0][1].shape))

# Structuring data
stages = [('train', data[0]), ('valid', data[1]), ('test', data[2])]
for stage, stage_data in stages:
    stage_folder = './data/%s' % stage
    if not os.path.isdir(stage_folder):
        print 'creating %s folder' % stage
        os.mkdir(stage_folder)
    stage_examples, stage_labels = stage_data
    print 'writing %s images to disk' % stage
    caffe_file = open('./data/%s.txt' % stage, 'w')
    for count, feats in enumerate(stage_examples):
        image = feats.reshape((28, 28)) * 255
        cv2.imwrite('./data/%s/%s_%s.jpg' % (stage, str(stage_labels[count]), str(count)), image)
        caffe_file.write('./%s/%s_%s.jpg %s\n' % (stage, str(stage_labels[count]), str(count), str(stage_labels[count])))
    caffe_file.close()

# View the images
os.system('nautilus data/test')
