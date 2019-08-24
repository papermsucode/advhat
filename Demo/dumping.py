import skimage.io as io
import os
import numpy as np
from tqdm import tqdm
import sys
import argparse

def main(args):
	
	# Output dirs creation
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	images = []
	labels = []
	label = 0
	for path in sorted(os.listdir(args.input_dir)):
		for name in sorted(os.listdir(os.path.join(args.input_dir,path))):
			if args.mx:
				images.append([[label],os.path.join(args.input_dir,path,name)])
			else:
				images.append(os.path.join(args.input_dir,path,name))
			labels.append(label)
		label += 1
	
	
	if args.mx:
		# MXnet model
		import mxnet as mx
		sym, arg_params, aux_params = mx.model.load_checkpoint(args.model, 0)
		sym = sym.get_internals()['fc1_output']
		model = mx.mod.Module(symbol=sym, context=mx.gpu(0), label_names = None)
		model.bind(data_shapes=[('data', (1, 3, 112, 112))])
		model.set_params(arg_params, aux_params)
		iterator = mx.image.ImageIter(batch_size=args.batch,data_shape=(3,112,112),imglist=images,path_root='')
	else:
		# TensorFlow model
		import tensorflow as tf
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
			sess = tf.Session(graph=graph)
			inp_place = tf.placeholder(np.array(['1','2'],dtype='str').dtype)
			pipeline = tf.data.Dataset.from_tensor_slices(inp_place)
			def parse(filename):
				image_string = tf.read_file(filename)
				image = tf.image.decode_jpeg(image_string,dct_method="INTEGER_ACCURATE")
				image = tf.cast(image,tf.float32)
				image = (image - 127.5)*0.0078125
				image = tf.transpose(image,perm=[2,0,1])
				return image
			pipeline = pipeline.map(parse,num_parallel_calls=4)
			pipeline = pipeline.batch(args.batch)
			pipiline = pipeline.prefetch(8)
			iterator = pipeline.make_initializable_iterator()
			next_element = iterator.get_next()
		sess.run(iterator.initializer,feed_dict={inp_place:images})				  
    
    # Embeddings evaluation
	embs = np.zeros((len(images),512),dtype=np.float32)
	for i in tqdm(range(int(np.ceil(len(images)/args.batch)))):
		if args.mx:
			db = mx.io.DataBatch(data=iterator.next().data)
			model.forward(db, is_train=False)
			emb = model.get_outputs()[0].asnumpy()
			length = min(args.batch,len(images)-i*args.batch)
			embs[i*args.batch:i*args.batch+length] = emb[:length]/np.expand_dims(np.sqrt(np.sum(emb[:length]**2,1)),1)
		else:
			db = sess.run(next_element)
			embs[i*args.batch:min((i+1)*args.batch,len(images))] = sess.run(embedding,feed_dict=\
										{image_input:db,keep_prob:1.0,is_train:False})
	
	# Centroids preparation
	anchor = np.zeros((label,512),dtype=np.float32)
	labels = np.array(labels)
	for i in range(label):
		tmp = np.sum(embs[labels==i],axis=0)
		anchor[i] = tmp/np.sqrt(np.sum(tmp**2))
	np.save(os.path.join(args.output_dir,'centroids'),anchor)
	names = open(os.path.join(args.output_dir,'centroids_names.txt'),'w')
	for i in sorted(os.listdir(args.input_dir)):
		names.write(i+'\n')
	names.close()            


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str, help='Directory with aligned images.')
    parser.add_argument('output_dir', type=str, help='Directory to save embeddings.')
    parser.add_argument('model',type=str, help='Path to the model.')
    parser.add_argument('--mx',action='store_true', help='Flag to use the original mxnet model.')
    parser.add_argument('--batch',type=int, help='Batch size.',default=30)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
