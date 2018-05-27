"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import random
from time import sleep
import pickle


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y 

def crop(image, random_crop, image_size):
    if image.shape[1]>image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return image
  
def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

#A small part of code from MTCNN, github: https://github.com/AITTSMD/MTCNN-Tensorflow
def extract_face_from_image(args):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor


    image_path = args.image_path
    try:
        img = misc.imread(image_path)
    except (IOError, ValueError, IndexError) as e:
        errorMessage = '{}: {}'.format(image_path, e)
        print(errorMessage)
    else:
        if img.ndim<2:
            print('Unable to align "%s"' % image_path)
            text_file.write('%s\n' % (output_filename))
            exit(0)
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        img = img[:,:,0:3]

        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces>0:
            det = bounding_boxes[:,0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces>1:

                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                img_center = img_size / 2
                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                det_arr.append(det[index,:])
            else:
                det_arr.append(np.squeeze(det))

            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-22, 0)
                bb[1] = np.maximum(det[1]-22, 0)
                bb[2] = np.minimum(det[2]+22, img_size[1])
                bb[3] = np.minimum(det[3]+22, img_size[0])
                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                scaled = misc.imresize(cropped, (182, 182), interp='bilinear')
        else:
            print('Unable to align "%s"' % image_path)
            text_file.write('%s\n' % (output_filename))


        images = np.zeros((1, 160, 160, 3))
        img = scaled
        if img.ndim == 2:
            img = to_rgb(img)
        

        img = prewhiten(img)
        img = crop(img, False, 160)
        img = flip(img, False)
        images[0,:,:,:] = img


    return images


#A small part of code from FaceNet, github: https://github.com/davidsandberg/facenet
def get_embedding_vector(args, images):
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            np.random.seed()
            


            facenet.load_model(args.model)
            
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            emb_array = np.zeros((1, embedding_size))
            


            feed_dict = { images_placeholder:images, phase_train_placeholder:False }
            emb_array = sess.run(embeddings, feed_dict=feed_dict)
    return emb_array


#Computing the distance from all vectors with the give face feature map
def compute_distances(args, emb_array):
    
    embedding_path = args.feature_map
            
    with open(embedding_path, 'rb') as f:
        face_array = pickle.load(f)



    distance = np.array([i[1] for i in face_array])

    distance = distance - np.array(emb_array[0])

    distance = np.sum(distance**2, axis=1)


    sorted_distance = np.argsort(distance)

    possibility_array = []

    index = 0

    while(distance[sorted_distance[index]] < 0.5):
        possibility_array.append(face_array[sorted_distance[index]][0])
        index += 1
    

    # name_index = np.argmin(distance)

    # print("gg", name_index)
    return possibility_array, face_array[sorted_distance[0]][0]


def main(args):

    images = extract_face_from_image(args)
    
    emb_array = get_embedding_vector(args, images)
            

    names, looks_like = compute_distances(args, emb_array)


    if len(names) == 0:    
        print("Sorry, we could not recogonize this person! Maybe you can try another one")
        print("However, she/he looks like:", looks_like)
    else:
        print("This person can be:")

        for index, name in enumerate(names):
            print(index+1, " : ", name)
    




def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image_path', type=str,
        help='Image size (height, width) in pixels.', default='/home/zrji/3314/jason.jpg')


    parser.add_argument('--model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file', default='/home/zrji/model')

    parser.add_argument('--feature_map', type=str, 
        help='The feature map of all faces, embedded as a vector of size 512', default='./feature_map/good_face')


    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
