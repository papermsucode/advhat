import argparse
import numpy as np
import cv2
from skimage import transform as trans
import tensorflow as tf
import os
import skimage.io as io
import sys
from tqdm import tqdm

import align.detect_face as detect_face

# Transform grey image to RGB image
def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

# Align face as ArcFace template 
def preprocess(img, landmark):
    image_size = [112,112]
    src = np.array([
		[38.2946, 51.6963],
		[73.5318, 51.5014],
		[56.0252, 71.7366],
		[41.5493, 92.3655],
		[70.7299, 92.2041] ], dtype=np.float32)
    dst = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]

    warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
    return warped

def main(args):

	# MTCNN
	with tf.Graph().as_default():
		sess = tf.Session()
		with sess.as_default():
			pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
	threshold = [ 0.6, 0.7, 0.7 ]
	factor = 0.709
    
    # Output dirs creation
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	images = []
	for path in sorted(os.listdir(args.input_dir)):
		if not os.path.exists(os.path.join(args.output_dir,path)):
			os.mkdir(os.path.join(args.output_dir,path))
		for name in sorted(os.listdir(os.path.join(args.input_dir,path))):
			images.append(os.path.join(path,name))

	# Alignment procedure
	for path in tqdm(images):
		img = io.imread(os.path.join(args.input_dir,path))
		if img.ndim == 2:
			img = to_rgb(img)
		img = img[:,:,0:3]
		_minsize = min(min(img.shape[0]//5, img.shape[1]//5),80)
		bounding_boxes, points = detect_face.detect_face(img, _minsize, pnet, rnet, onet, threshold, factor)
		if bounding_boxes.size>0:
			bindex = -1
			nrof_faces = bounding_boxes.shape[0]
			if nrof_faces>0:
				det = bounding_boxes[:,0:4]
				img_size = np.asarray(img.shape)[0:2]
				bindex = 0
				if nrof_faces>1:				
					bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
					img_center = img_size / 2
					offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
					offset_dist_squared = np.sum(np.power(offsets,2.0),0)
					bindex = np.argmax(bounding_box_size-offset_dist_squared*2.0)
			points = points[:, bindex]
			landmark = points.reshape((2,5)).T
			warped = preprocess(img, landmark)
			io.imsave(os.path.join(args.output_dir,path), warped)
		else:
			print(path+' was skipped')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory for aligned face thumbnails.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
