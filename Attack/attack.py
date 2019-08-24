import argparse
import sys
import os
import tensorflow as tf
import numpy as np
import skimage.io as io
from skimage.transform import rescale
from tqdm import tqdm
from stn import spatial_transformer_network as stn
from utils import TVloss, projector
from sklearn.linear_model import LinearRegression as LR
from time import time
import datetime
import matplotlib.pyplot as plt

# Prepare image to network input format
def prep(im):
    if len(im.shape)==3:
        return np.transpose(im,[2,0,1]).reshape((1,3,112,112))*2-1
    elif len(im.shape)==4:
        return np.transpose(im,[0,3,1,2]).reshape((im.shape[0],3,112,112))*2-1

def main(args):
        print(args)
        now = str(datetime.datetime.now())
        
        sess = tf.Session()
        
        # Off-plane sticker projection
        logo = tf.placeholder(tf.float32,shape=[None,400,900,3],name='logo_input')
        param = tf.placeholder(tf.float32,shape=[None,1],name='param_input')
        ph = tf.placeholder(tf.float32,shape=[None,1],name='ph_input')
        result = projector(param,ph,logo)

        # Union of the sticker and face image
        mask_input = tf.placeholder(tf.float32,shape=[None,900,900,3],name='mask_input')
        face_input = tf.placeholder(tf.float32,shape=[None,600,600,3],name='face_input')
        theta = tf.placeholder(tf.float32,shape=[None,6],name='theta_input')
        prepared = stn(result,theta)
		
		# Transformation to ArcFace template
        theta2 = tf.placeholder(tf.float32,shape=[None,6],name='theta2_input')
        united = prepared[:,300:,150:750]*mask_input[:,300:,150:750]+\
                                        face_input*(1-mask_input[:,300:,150:750])
        final_crop = tf.clip_by_value(stn(united,theta2,(112,112)),0.,1.)
        
        # TV loss and gradients
        w_tv = tf.placeholder(tf.float32,name='w_tv_input')
        tv_loss = TVloss(logo,w_tv)

        grads_tv = tf.gradients(tv_loss,logo)
        grads_input = tf.placeholder(tf.float32,shape=[None,112,112,3],name='grads_input')
        grads1 = tf.gradients(final_crop,logo,grad_ys=grads_input)
        
        # Varios images generator
        class Imgen(object):
                def __init__(self):
                        self.fdict = {ph:[[args.ph]],\
                                                  logo:np.ones((1,400,900,3)),\
                                                  param:[[args.param]],\
                                                  theta:1./args.scale*np.array([[1.,0.,-args.x/450.,0.,1.,-args.y/450.]]),\
                                                  theta2:[[1.,0.,0.,0.,1.,0.]],\
                                                  w_tv:args.w_tv}
                        mask = sess.run(prepared,feed_dict=self.fdict)
                        self.fdict[mask_input] = mask
                        
                def gen_fixed(self,im,advhat):
                        self.fdict[face_input] = np.expand_dims(im,0)
                        self.fdict[logo] = np.expand_dims(advhat,0)
                        return self.fdict, sess.run(final_crop,feed_dict=self.fdict)
                
                def gen_random(self,im,advhat,batch=args.batch_size):
                        alpha1 = np.random.uniform(-1.,1.,size=(batch,1))/180.*np.pi
                        scale1 = np.random.uniform(args.scale-0.02,args.scale+0.02,size=(batch,1))
                        y1 = np.random.uniform(args.y-600./112.,args.y+600./112.,size=(batch,1))
                        x1 = np.random.uniform(args.x-600./112.,args.x+600./112.,size=(batch,1))
                        alpha2 = np.random.uniform(-1.,1.,size=(batch,1))/180.*np.pi
                        scale2 = np.random.uniform(1./1.04,1.04,size=(batch,1))
                        y2 = np.random.uniform(-1.,1.,size=(batch,1))/66.
                        angle = np.random.uniform(args.ph-2.,args.ph+2.,size=(batch,1))
                        parab = np.random.uniform(args.param-0.0002,args.param+0.0002,size=(batch,1))
                        fdict = {ph:angle,param:parab,w_tv:args.w_tv,\
                                        theta:1./scale1*np.hstack([np.cos(alpha1),np.sin(alpha1),-x1/450.,\
                                                                                           -np.sin(alpha1),np.cos(alpha1),-y1/450.]),\
                                        theta2:scale2*np.hstack([np.cos(alpha2),np.sin(alpha2),np.zeros((batch,1)),\
                                                                                        -np.sin(alpha2),np.cos(alpha2),y2]),\
                                        logo:np.ones((batch,400,900,3)),\
                                        face_input:np.tile(np.expand_dims(im,0),[batch,1,1,1])}
                        mask = sess.run(prepared,feed_dict=fdict)
                        fdict[mask_input] = mask
                        fdict[logo] = np.tile(np.expand_dims(advhat,0),[batch,1,1,1])
                        return fdict, sess.run(final_crop,feed_dict=fdict)
                        
        gener = Imgen()

        # Initialization of the sticker
        init_logo = np.ones((400,900,3))*127./255.
        if args.init_face!=None:
                init_face = io.imread(args.init_face)/255.
                init_loss = tv_loss+tf.reduce_sum(tf.abs(init_face-united[0]))
                init_grads = tf.gradients(init_loss,logo)
                init_logo = np.ones((400,900,3))*127./255.
                fdict, _ = gener.gen_fixed(init_face,init_logo)
                moments = np.zeros((400,900,3))
                print('Initialization from face, step 1/2')
                for i in tqdm(range(500)):
                        fdict[logo] = np.expand_dims(init_logo,0)
                        grads = moments*0.9+sess.run(init_grads,feed_dict=fdict)[0][0]
                        moments = moments*0.9 + grads*0.1
                        init_logo = np.clip(init_logo-1./51.*np.sign(grads),0.,1.)
                print('Initialization from face, step 2/2')
                for i in tqdm(range(500)):
                        fdict[logo] = np.expand_dims(init_logo,0)
                        grads = moments*0.9+sess.run(init_grads,feed_dict=fdict)[0][0]
                        moments = moments*0.9 + grads*0.1
                        init_logo = np.clip(init_logo-1./255.*np.sign(grads),0.,1.)
                io.imsave(now+'_init_logo.png',init_logo)
        elif args.init_logo!=None:
                init_logo[:] = io.imread(args.init_logo)/255.
                
                
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

        orig_emb = tf.placeholder(tf.float32,shape=[None,512],name='orig_emb_input')
        cos_loss = tf.reduce_sum(tf.multiply(embedding,orig_emb),axis=1)
        grads2 = tf.gradients(cos_loss,image_input)

        fdict2 = {keep_prob:1.0,is_train:False}
        
        # Anchor embedding calculation
        if args.anchor_face!=None:
                anch_im = rescale(io.imread(args.anchor_face)/255.,112./600.,order=5)
                fdict2[image_input] = prep(anch_im)
                fdict2[orig_emb] = sess.run(embedding,feed_dict=fdict2)
        elif args.anchor_emb!=None:
                fdict2[orig_emb] = np.load(args.anchor_emb)[-1:]
        else:
                anch_im = rescale(io.imread(args.image)/255.,112./600.,order=5)
                fdict2[orig_emb] = sess.run(embedding,feed_dict=fdict2)
        
        # Attack constants
        im0 = io.imread(args.image)/255.
        regr = LR(n_jobs=4)
        regr_len = 100
        regr_coef = -1.
        moments = np.zeros((400,900,3))
        moment_val = 0.9
        step_val = 1./51.
        stage = 1
        step = 0
        lr_thresh = 100
        ls = []
        t = time()
        while True:
                # Projecting sticker to the face and feeding it to the embedding model
                fdict,ims = gener.gen_random(im0,init_logo)
                fdict2[image_input] = prep(ims)
                grad_tmp = sess.run(grads2,feed_dict=fdict2)
                
                fdict_val, im_val = gener.gen_fixed(im0,init_logo)
                fdict2[image_input] = prep(im_val)
                ls.append(sess.run(cos_loss,feed_dict=fdict2)[0])
                
                # Gradients to the original sticker image
                fdict[grads_input] = np.transpose(grad_tmp[0],[0,2,3,1])
                grads_on_logo = np.mean(sess.run(grads1,feed_dict=fdict)[0],0)
                grads_on_logo += sess.run(grads_tv,feed_dict=fdict)[0][0]
                moments = moments*moment_val + grads_on_logo*(1.-moment_val)
                init_logo -= step_val*np.sign(moments)
                init_logo = np.clip(init_logo,0.,1.)
                
                # Logging
                step += 1
                if step%20==0:
                        print('Stage:',stage,'Step:',step,'Av. time:',round((time()-t)/step,2),'Loss:',round(ls[-1],2),'Coef:',regr_coef)

                # Switching to the second stage
                if step>lr_thresh:
                        regr.fit(np.expand_dims(np.arange(100),1),np.hstack(ls[-100:]))
                        regr_coef = regr.coef_[0]
                        if regr_coef>=0:
                                if stage==1:
                                        stage = 2
                                        moment_val = 0.995
                                        step_val = 1./255.
                                        step = 0
                                        regr_coef = -1.
                                        lr_thresh = 200
                                        t = time()
                                else:
                                        break

        plt.plot(range(len(ls)),ls)
        plt.savefig(now+'_cosine.png')
        io.imsave(now+'_advhat.png',init_logo)
                   
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image', type=str, help='Path to the image for attack.')
    parser.add_argument('model', type=str, help='Path to the model for attack.')
    parser.add_argument('--init_face', type=str, default=None, help='Path to the face for sticker inititalization.')
    parser.add_argument('--init_logo', type=str, default=None, help='Path to the image for inititalization.')
    parser.add_argument('--anchor_face', type=str, default=None, help='Path to the anchor face.')
    parser.add_argument('--anchor_emb', type=str, default=None, help='Path to the anchor emb (the last will be used)')
    parser.add_argument('--w_tv', type=float, default=1e-4, help='Weight of the TV loss')
    parser.add_argument('--ph', type=float, default=17., help='Angle of the off-plane rotation')
    parser.add_argument('--param', type=float, default=0.0013, help='Parabola rate for the off-plane parabolic transformation')
    parser.add_argument('--scale', type=float, default=0.465, help='Scaling parameter for the sticker')
    parser.add_argument('--x', type=float, default=0., help='Translation of the sticker along x-axis')
    parser.add_argument('--y', type=float, default=-15., help='Translation of the sticker along y-axis')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for attack')
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
