import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf
from align import detect_face
from skimage import transform as trans
from skimage.io import imsave
import os
import datetime

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
	
	# Models download
	frozen_graph = args.model
	with tf.gfile.GFile(frozen_graph, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
	with tf.Graph().as_default() as graph:
		  tf.import_graph_def(graph_def,
							  input_map=None,
							  return_elements=None,
							  name="")
		  image_input = graph.get_tensor_by_name('image_input:0')
		  keep_prob = graph.get_tensor_by_name('keep_prob:0')
		  is_train = graph.get_tensor_by_name('training_mode:0')
		  embedding = graph.get_tensor_by_name('embedding:0')

		  minsize = 100
		  threshold = [ 0.6, 0.7, 0.7 ] 
		  factor = 0.709
		  sess = tf.Session(graph=graph)
		  pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
	
	# Centroids download
	anchor = np.load(os.path.join(args.centroids,'centroids.npy'))
	names = open(os.path.join(args.centroids,'centroids_names.txt')).read().split('\n')[:-1]

	IDcolor = [255., 255., 255.]
	IDcolor2 = [255., 0., 0.]

	video_capture = cv2.VideoCapture(0)
	video_capture.set(3, 1280)
	video_capture.set(4, 1024)

	while(True):
		
		# Start of video sequence processing
		ret, frame = video_capture.read()
		frame = cv2.flip(frame[:,:,::-1], 1)
		if not ret:
			print('Cannot access the webcam')
			break
		
		key = cv2.waitKey(1)			
		if key == ord('q'):
			break
		if key == ord('s'):
			imsave('Demo-'+str(datetime.datetime.now())+'.jpg',frame)
			
		# Search and preparation of all faces on the frame
		bounding_boxes, points = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
		
		batch = np.zeros((bounding_boxes.shape[0],3,112,112),dtype=np.float32)    
		for i in range(bounding_boxes.shape[0]):
			landmark = points[:,i].reshape((2,5)).T
			warped = preprocess(frame, landmark = landmark)
			warped = np.transpose(warped,[2,0,1]).reshape((1,3,112,112))
			batch[i] = (warped-127.5)*0.0078125
		
		# Recognition of all faces
		if batch.shape[0]!=0:
			embs = sess.run(embedding,feed_dict={image_input:batch,keep_prob:1.0,is_train:False})
			for i in range(bounding_boxes.shape[0]):
				probabilities = np.dot(anchor,embs[i])
				val = np.max(probabilities)
				pos = np.argmax(probabilities)

				pt1 = (int(bounding_boxes[i][0]), int(bounding_boxes[i][1]))
				pt2 = (int(bounding_boxes[i][2]), int(bounding_boxes[i][3]))

				cv2.rectangle(frame, pt1, pt2, IDcolor)
			
				cv2.putText(frame, 'Top-1 class: '+names[pos],
								(int(bounding_boxes[i][0]), int(bounding_boxes[i][1])-5),
								cv2.FONT_HERSHEY_SIMPLEX, 1., IDcolor, 3)
				cv2.putText(frame, 'Sim. to top-1 class: '+str(round(val,4)),
								(int(bounding_boxes[i][0]), int(bounding_boxes[i][3])+30),
								cv2.FONT_HERSHEY_SIMPLEX, 1., IDcolor, 3)

		cv2.imshow('Camera ("q" to quit, "s" to save frame)', frame[:,:,::-1])
		
				
	video_capture.release()
	cv2.destroyAllWindows()
	
	

def parse_arguments(argv):
	parser = argparse.ArgumentParser()
    
	parser.add_argument('model',type=str, help='Path to the model.')
	parser.add_argument('centroids',type=str, help='Dir with centoids of classes for classifier.')
	return parser.parse_args(argv)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
