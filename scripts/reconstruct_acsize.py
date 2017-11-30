#!/usr/bin/env python
import argparse


from blocks.serialization import load
from matplotlib import cm, pyplot
from mpl_toolkits.axes_grid1 import ImageGrid
from theano import tensor
import numpy as np
import scipy
import scipy.misc
import theano

from ali import streams


def main(main_loop, data_stream, nrows, ncols, save_path):
    ali, = main_loop.model.top_bricks
    x = tensor.tensor4('features')
    examples, = next(data_stream.get_epoch_iterator())
    reconstructions = theano.function([x], ali.reconstruct(x))(examples)

    #figure = pyplot.figure()
    #grid = ImageGrid(figure, 111, (nrows, 2 * ncols), axes_pad=0.01,)
    #images = numpy.empty(
     #   (2 * nrows * ncols,) + examples.shape[1:], dtype=examples.dtype)
    #images[::2] = examples
    #images[1::2] = 2*reconstructions-1

    samples=2*reconstructions-1
    examples=2*examples-1
    samples = np.transpose(samples[:nrows*ncols, ], (0, 2, 3, 1))
    examples=np.transpose(examples[:nrows*ncols, ], (0, 2, 3, 1))
    samples = [samples[i, :, :, :] for i in range(nrows*ncols)]
    exampless=[examples[i,:,:,:] for i in range(nrows*ncols) ]
    rows = []
    for i in range(nrows):
        rows.append(np.concatenate(exampless[i::ncols],0))
        rows.append(np.concatenate(samples[i::ncols], 0))
        #print(rows[-1].shape)

        #print(rows[-1].shape)
    samples = np.concatenate(rows, 1)
    scipy.misc.imsave(save_path, samples)




if __name__ == "__main__":
    stream_functions = {
        'cifar10': streams.create_cifar10_data_streams,
        'svhn': streams.create_svhn_data_streams,
        'celeba': streams.create_celeba_data_streams,
        'tiny_imagenet': streams.create_tiny_imagenet_data_streams}
    parser = argparse.ArgumentParser(description="Plot reconstructions.")
    parser.add_argument("which_dataset", type=str,
                        choices=tuple(stream_functions.keys()),
                        help="which dataset to compute reconstructions on.")
    parser.add_argument("main_loop_path", type=str,
                        help="path to the pickled main loop.")
    parser.add_argument("--nrows", type=int, default=4,
                        help="number of rows of samples to display.")
    parser.add_argument("--ncols", type=int, default=10,
                        help="number of columns of samples to display.")
    parser.add_argument("--save-path", type=str, default=None,
                        help="where to save the reconstructions.")
    args = parser.parse_args()

    with open(args.main_loop_path, 'rb') as src:
        main_loop = load(src)
    num_examples = args.nrows * args.ncols
    rng = np.random.RandomState()
    _1, _2, data_stream = stream_functions[args.which_dataset](num_examples,
                                                               num_examples,
                                                               rng=rng)
    main(main_loop, data_stream, args.nrows, args.ncols, args.save_path)

