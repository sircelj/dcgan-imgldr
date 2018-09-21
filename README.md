# DCGAN with an ImageLoader
Tensorflow implementation of a [DCGAN](https://arxiv.org/abs/1511.06434) I wrote for a school project. The implementation uses a ImageLoader which is responsible for feeding the model with learning data during learning and can be repurposed for differend kinds od data. Here I used it for the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) data set and for the [speech commands dataset](https://arxiv.org/abs/1804.03209) where the audio is transformed into an SFTF image representation.

## ImageLoader usage
The `ImageLoader` is originally implemented to give cropped images of the CelebA dataset, but you can write a subclass that can essentially work with any kind of data, as long as the new subclass is going to return data that is in a 3D shape.

To repurpose it, create a new class with the `ImageLoader` parrent. The main method that probably would need to get overridden is `_get_image()`, which transforms the original data into a desired 3D shape.

For an example see the `SCC` class, which is an example of an `ImageLoader` subclass that reads audio data and constructs a polar form [STFT](https://en.wikipedia.org/wiki/Short-time_Fourier_transform) which can be modeled by the DCGAN.

## Needed libraries
- Python 3.X
- [Tensorflow](https://www.tensorflow.org/install/)
- [Numpy/Scipy](https://www.scipy.org/install.html)
- [Matplotlib](https://matplotlib.org/users/installing.html)
