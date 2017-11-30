# LSAL_generator_classifier
This is the code for Loss Sensitive Adversarial Learning with Manifold Margins. The base of the code has been borrowed from ALI [[ALI](https://github.com/IshmaelBelghazi/ALI/)]
## Requirements

* [Blocks](https://blocks.readthedocs.org/en/latest/), development version
* [Fuel](https://fuel.readthedocs.org/en/latest/), development version
## Setup

Clone the repository, then install with

``` bash
$ pip install -e ALI
```

## Downloading and converting the datasets

Set up your `~/.fuelrc` file:

``` bash
$ echo "data_path: \"<MY_DATA_PATH>\"" > ~/.fuelrc
```

Go to `<MY_DATA_PATH>`:

``` bash
$ cd <MY_DATA_PATH>
```

Download the CIFAR-10 dataset:

``` bash
$ fuel-download cifar10
$ fuel-convert cifar10
$ fuel-download cifar10 --clear
```

Download the SVHN format 2 dataset:

``` bash
$ fuel-download svhn 2
$ fuel-convert svhn 2
$ fuel-download svhn 2 --clear
```

Download the CelebA dataset:

``` bash
$ fuel-download celeba 64
$ fuel-convert celeba 64
$ fuel-download celeba 64 --clear
```

## Training the models
Make sure you're in the repo's root directory.

### CIFAR-10

``` bash
$ THEANORC=theanorc python experiments/LSAL_cifar10.py
```

### SVHN

``` bash
$ THEANORC=theanorc python experiments/LSAL_svhn.py
```

### CelebA

``` bash
$ THEANORC=theanorc python experiments/LSAL_celeba_savemargins.py
```

## Evaluating the models

### Samples

``` bash
$ THEANORC=theanorc scripts/sample [main_loop.tar]
```

e.g.

``` bash
$ THEANORC=theanorc scripts/sample LSAL_cifar10.tar
```

### Interpolations

``` bash
$ THEANORC=theanorc scripts/interpolate [which_dataset] [main_loop.tar]
```

e.g.

``` bash
$ THEANORC=theanorc scripts/interpolate celeba LSAL_celeba.tar
```

### Reconstructions

``` bash
$ THEANORC=theanorc scripts/reconstruct [which_dataset] [main_loop.tar]
```

e.g.

``` bash
$ THEANORC=theanorc scripts/reconstruct cifar10 LSAL_cifar10.tar
```
