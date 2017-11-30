#!/usr/bin/env python
import argparse
import numpy as np
import scipy
import scipy.misc
import theano
from blocks.serialization import load
#from matplotlib import cm, pyplot
#from mpl_toolkits.axes_grid1 import ImageGrid


def main(main_loop, nrows, ncols, save_path=None):
    ali, = main_loop.model.top_bricks
    input_shape = ali.encoder.get_dim('output')
    input_shape=(input_shape[0],input_shape[1],input_shape[2])
    #input_shape=(input_shape[0]*2,input_shape[1],input_shape[2])

    z = ali.theano_rng.normal(size=(nrows * ncols,) + input_shape)
    tmp=z.eval()
    #print('shape',tmp.shape)
    x = ali.sample(z)
    samples = theano.function([], x)()
    samples=2*samples-1
    samples = np.transpose(samples[:nrows*ncols, ], (0, 2, 3, 1))
    samples = [samples[i, :, :, :] for i in range(nrows*ncols)]
    rows = []
    for i in range(nrows):
        rows.append(np.concatenate(samples[i::ncols], 1))
    samples = np.concatenate(rows, 0)
    scipy.misc.imsave(save_path, samples)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot samples.")
    parser.add_argument("main_loop_path", type=str,
                        help="path to the pickled main loop.")
    parser.add_argument("--nrows", type=int, default=10,
                        help="number of rows of samples to display.")
    parser.add_argument("--ncols", type=int, default=10,
                        help="number of columns of samples to display.")
    parser.add_argument("--save-path", type=str, default=None,
                        help="where to save the generated samples.")
    args = parser.parse_args()

    with open(args.main_loop_path, 'rb') as src:
        main(load(src), args.nrows, args.ncols, args.save_path)

