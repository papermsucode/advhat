import numpy as np
import tensorflow as tf


def tf_integral(x,a):
	return 0.5*(x*tf.sqrt(x**2+a)+a*tf.log(tf.abs(x+tf.sqrt(x**2+a))))
def tf_pre_parabol(x,par):
	x = x-450.
	prev = 2.*par*(tf_integral(tf.abs(x),0.25/(par**2))-tf_integral(0,0.25/(par**2)))
	return prev+450.
	
def projector(param,ph,logo):
	'''Apply off-plane transformations to the sticker images
	param: parabola rate of the off-plane parabolic tranformation, rank 2 tensor with shape [N, 1]
	ph:angle of the off-plane rotation, rank 2 tensor with shape [N, 1]
	logo: rank 4 tensor with format NHWC and shape [N, 400, 900, 3]
	
	return: rank 4 tensor with format NHWC and shape [N, 900, 900, 3]
	'''
	right_cumsum = tf.transpose(tf.pad(tf.cumsum(logo[:,:,450:],axis=2),tf.constant([[0,0],[0,0],[1,0],[0,0]])),[0,2,1,3])
	left_cumsum = tf.transpose(tf.pad(tf.cumsum(logo[:,:,:450][:,:,::-1],axis=2),tf.constant([[0,0],[0,0],[1,0],[0,0]])),[0,2,1,3])
	
	anchors = tf.expand_dims(tf.cast(tf.round(tf.clip_by_value(\
				tf_pre_parabol(tf.expand_dims(tf.constant(np.arange(450,901,dtype=np.float32)),0),\
				  param)-450.,0,450.)),tf.int32),2)
	anch_inds = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(tf.shape(param)[0]),1),2),[1,451,1])
	new_anchors = tf.concat([anch_inds,anchors],2)
	
	anchors_div = tf.expand_dims(tf.cast(tf.clip_by_value(anchors[:,1:]-anchors[:,:-1],1,900),tf.float32),3)
	right_anchors_cumsum = tf.gather_nd(right_cumsum,new_anchors)
	right_anchors_diffs = right_anchors_cumsum[:,1:]-right_anchors_cumsum[:,:-1]
	right = right_anchors_diffs/anchors_div
	left_anchors_cumsum = tf.gather_nd(left_cumsum,new_anchors)
	left_anchors_diffs = left_anchors_cumsum[:,1:]-left_anchors_cumsum[:,:-1]
	left = left_anchors_diffs/anchors_div
	
	tmp_result = tf.transpose(tf.concat([left[:,::-1],right],axis=1),[0,2,1,3])
	
	cumsum = tf.pad(tf.cumsum(tmp_result,axis=1),tf.constant([[0,0],[1,0],[0,0],[0,0]]))
	
	angle = tf.expand_dims(np.pi/180.*ph,2)
	
	z = param*tf.constant((np.arange(900,dtype=np.float32)-449.5)**2)
	z_tile = tf.tile(tf.expand_dims(z,1),tf.constant([1,901,1]))
	
	y_coord = tf.constant(np.arange(-250,651,dtype=np.float32))
	y_tile = tf.tile(tf.expand_dims(tf.expand_dims(y_coord,1),0),[tf.shape(param)[0],1,900])
	
	y_prev = (y_tile+z_tile*tf.sin(-angle))/tf.cos(angle)
	y_round = tf.cast(tf.round(tf.clip_by_value(y_prev,0,400.)),tf.int32)
	y_div = tf.clip_by_value(y_round[:,1:]-y_round[:,:-1],1,900)
	
	x_coord = tf.constant(np.arange(900,dtype=np.int32))
	x_tile = tf.tile(tf.expand_dims(tf.expand_dims(x_coord,0),0),[tf.shape(param)[0],901,1])
	
	b_coord = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(tf.shape(param)[0]),1),2),[1,901,900])
	
	indices = tf.stack([b_coord,y_round,x_tile],axis=3)
	
	chosen_cumsum = tf.gather_nd(cumsum,indices)
	chosen_cumsum_diffs = chosen_cumsum[:,1:]-chosen_cumsum[:,:-1]
	final_results = tf.clip_by_value(chosen_cumsum_diffs/tf.expand_dims(tf.cast(y_div,tf.float32),3),0.,1.)
	
	return final_results

def TVloss(logo,w_tv):
	'''Calculate TV loss of the sticker image with predefined weight.
	logo: rank 4 tensor with format NHWC
	w_tv: weight of the TV loss
	
	return: scalar value of the TV loss
	'''
	vert_diff = logo[:,1:]-logo[:,:-1]
	hor_diff = logo[:,:,1:]-logo[:,:,:-1]
	vert_diff_sq = tf.square(vert_diff)
	hor_diff_sq = tf.square(hor_diff)
	vert_pad = tf.pad(vert_diff_sq,tf.constant([[0,0],[1,0],[0,0],[0,0]]))
	hor_pad = tf.pad(hor_diff_sq,tf.constant([[0,0],[0,0],[1,0],[0,0]]))
	tv_sum = vert_pad+hor_pad
	tv = tf.sqrt(tv_sum+1e-5)
	tv_final_sum = tf.reduce_sum(tv)
	tv_loss = w_tv*tv_final_sum
	return tv_loss
