#!/usr/bin/env python
import argparse
import numpy as np
import scipy
import scipy.misc
import theano
import theano.tensor as T

from blocks.serialization import load
#from matplotlib import cm, pyplot
#from mpl_toolkits.axes_grid1 import ImageGrid


def main(main_loop, nsamples, save_path=None):
    ali, = main_loop.model.top_bricks
    input_shape = ali.encoder.get_dim('output')
    input_shape=(input_shape[0],input_shape[1],input_shape[2])
    #input_shape=(input_shape[0]*2,input_shape[1],input_shape[2])
    z= T.tensor4()
    x = ali.sample(z)
    sample = theano.function([z], x)
    bsize=1000
    nbatch=np.int32(np.ceil(nsamples/bsize))
    print('nbatch_size:',nbatch)
    for i in range(nbatch):
      #  print("*****",i)
        z = np.float32(np.random.normal(0.0,1.0,size=(bsize,) + input_shape))
        print(z.shape)
        samples=sample(z)
        samples=2*samples-1
        samples = np.transpose(samples[:bsize, ], (0, 2, 3, 1))
        samples = [samples[k, :, :, :] for k in range(bsize)]
        for j in range(bsize):
            #print(i*bsize)
            #print(j)
            img_name='img_{}.png'.format(i*bsize+j)
            scipy.misc.imsave(save_path+'/'+img_name, samples[j])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot samples.")
    parser.add_argument("main_loop_path", type=str,
                        help="path to the pickled main loop.")
    parser.add_argument("--nsamples", type=int, default=50000,
                        help="number of  samples to generate.")
    parser.add_argument("--save-path", type=str, default=None,
                        help="where to save the generated samples.")
    args = parser.parse_args()

    with open(args.main_loop_path, 'rb') as src:
        main(load(src), args.nsamples, args.save_path)

