# glow-pytorch
PyTorch implementation of Glow, Generative Flow with Invertible 1x1 Convolutions (https://arxiv.org/abs/1807.03039)

Usage:

> python train.py PATH

as trainer uses ImageFolder of torchvision, input directory should be structured like this even when there are only 1 classes. (Currently this implementation does not incorporate class classification loss.)

> PATH/class1 <br/>
> PATH/class2 <br/>
> ...

## Notes

![Sample](sample.png)

I have trained model on vanilla celebA dataset. Seems like works well. I found that learning rate (I have used 1e-4 without scheduling), learnt prior, number of bits (in this cases, 5), and using sigmoid function at the affine coupling layer instead of exponential function is beneficial to training a model.

In my cases, LU decomposed invertible convolution was much faster than plain version. So I made it default to use LU decomposed version.

![Progression of samples](progression.gif)

Progression of samples during training. Sampled once per 100 iterations during training.