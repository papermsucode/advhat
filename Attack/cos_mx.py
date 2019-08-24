import argparse
import sys
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import skimage.io as io
from skimage.transform import rescale
from numpy import linalg as LA

# Prepare image to network input format
def prep(im):
    if len(im.shape)==3:
        return np.transpose(im,[2,0,1]).reshape((1,3,112,112))
    elif len(im.shape)==4:
        return np.transpose(im,[0,3,1,2]).reshape((im.shape[0],3,112,112))

def main(args):
        print(args)
        
        # Embedding model
        sym, arg_params, aux_params = mx.model.load_checkpoint(args.model, 0)
        sym = sym.get_internals()['fc1_output']
        model = mx.mod.Module(symbol=sym, context=mx.gpu(0), label_names = None)
        model.bind(data_shapes=[('data', (1, 3, 112, 112))])
        model.set_params(arg_params, aux_params)
        
        # Embedding calculation
        im1 = (prep(rescale(io.imread(args.face1)/255.,112./600.,order=5))*255.).astype(np.uint8)
        im2 = (prep(rescale(io.imread(args.face2)/255.,112./600.,order=5))*255.).astype(np.uint8)
        
        batch = mx.io.DataBatch(data=[nd.array(im1)])
        model.forward(batch, is_train=False)
        emb1 = model.get_outputs()[0].asnumpy()[0]
        batch = mx.io.DataBatch(data=[nd.array(im2)])
        model.forward(batch, is_train=False)
        emb2 = model.get_outputs()[0].asnumpy()[0]

        # Normalization
        emb1 /= LA.norm(emb1)
        emb2 /= LA.norm(emb2)
        cos_sim = np.sum(emb1 * emb2)

        # Result
        print('Cos_sim(face1, face2) =', cos_sim) 
                   
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('face1', type=str, help='Path to the preprocessed face1.')
    parser.add_argument('face2', type=str, help='Path to the preprocessed face2.')
    parser.add_argument('model', type=str, help='Path to the model.')
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
