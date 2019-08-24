import argparse
import sys
import tensorflow as tf
import numpy as np
import skimage.io as io
from skimage.transform import rescale

# Prepare image to network input format
def prep(im):
    if len(im.shape)==3:
        return np.transpose(im,[2,0,1]).reshape((1,3,112,112))*2-1
    elif len(im.shape)==4:
        return np.transpose(im,[0,3,1,2]).reshape((im.shape[0],3,112,112))*2-1

def main(args):
        print(args)
        
        sess = tf.Session()
        
        # Embedding model
        with tf.gfile.GFile(args.model, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def,
                                          input_map=None,
                                          return_elements=None,
                                          name="")
        image_input = tf.get_default_graph().get_tensor_by_name('image_input:0')
        keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')
        is_train = tf.get_default_graph().get_tensor_by_name('training_mode:0')
        embedding = tf.get_default_graph().get_tensor_by_name('embedding:0')

        tfdict = {keep_prob:1.0, is_train:False}
        
        # Embedding calculation
        im1 = prep(rescale(io.imread(args.face1)/255.,112./600.,order=5))
        im2 = prep(rescale(io.imread(args.face2)/255.,112./600.,order=5))
        tfdict[image_input] = im1
        emb1 = sess.run(embedding,feed_dict=tfdict)
        tfdict[image_input] = im2
        emb2 = sess.run(embedding,feed_dict=tfdict)

        # Result
        cos_sim = np.sum(emb1 * emb2)
        print('Cos_sim(face1, face2) =', cos_sim) 
                   
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('face1', type=str, help='Path to the preprocessed face1.')
    parser.add_argument('face2', type=str, help='Path to the preprocessed face2.')
    parser.add_argument('model', type=str, help='Path to the model.')
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
