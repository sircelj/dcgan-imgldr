import os
import numpy as np
from scipy.misc import imresize, imread
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plotimage(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples[:64]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(ImageLoader.tanh2rgb(sample.reshape(64, 64, 3)).astype(np.uint8), cmap='Greys_r')
    return fig


def plotimage_row(samples):
    n_samples = samples.shape[0]
    fig = plt.figure(figsize=(n_samples, 1))
    gs = gridspec.GridSpec(1, n_samples)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(ImageLoader.tanh2rgb(sample.reshape(64, 64, 3)).astype(np.uint8), cmap='Greys_r')
    plt.tight_layout()
    return fig

class ImageLoader:
    def __init__(self, im_dir, batch_size, start_epoch=0, im_width=64, im_height=64, im_c=3):
        self.im_dir = im_dir
        self.image_index = 0
        self.epoch = start_epoch
        self.batch_size = batch_size
        self.file_names = [name for name in os.listdir(im_dir)
                           if os.path.isfile(im_dir + name)]
        np.random.shuffle(self.file_names)
        self.number_of_images = len(self.file_names)
        self.im_width = im_width
        self.im_height = im_height
        self.im_c = im_c

    def __get_image(self, name):
        im = imread(self.im_dir + name)
        im_h, im_w, im_c = im.shape
        im = imresize(im[(im_h // 2 - 64):(im_h // 2 + 64), (im_w // 2 - 64):(im_w // 2 + 64), :],
                      [self.im_width, self.im_height])
        return self.rgb2tanh(im)

    def get_new_batch(self):
        batch = np.zeros([self.batch_size, self.im_width, self.im_height, self.im_c])
        for i, name in enumerate(self.file_names[self.image_index:self.image_index+self.batch_size]):
            im = self.__get_image(name)
            batch[i] = im
        self.image_index += self.batch_size

        if self.image_index + self.batch_size > self.number_of_images:
            self.epoch += 1
            self.image_index = 0
            np.random.shuffle(self.file_names)

        return batch

    @staticmethod
    def __transform_area(x, a, b, alpha, beta):
        return ((alpha - beta) * x + beta * a - alpha * b) / (a - b)

    @staticmethod
    def rgb2tanh(x):
        return ImageLoader.__transform_area(x, 0, 256, -1, 1)

    @staticmethod
    def rgb2sigmoid(x):
        return ImageLoader.__transform_area(x, 0, 256, 0, 1)

    @staticmethod
    def tanh2rgb(x):
        return ImageLoader.__transform_area(x, -1, 1, 0, 256)

    @staticmethod
    def sigmoid2rgb(x):
        return ImageLoader.__transform_area(x, 0, 1, 0, 256)


"""
class CelebA(ImageLoader):

    def __init__(self):
        super().__init__()
"""

if __name__ == "__main__":
    # il = ImageLoader("../img_align_celeba/", 64)
    il = ImageLoader("../img_small/", 64)
    for _ in range(10):
        print(il.epoch)
        batch = il.get_new_batch()
        plt.subplot(121)
        plt.imshow(il.tanh2rgb(batch[0]).astype(np.uint8))  # Convert to uint8 to get correct colours
        plt.subplot(122)
        plt.imshow(il.tanh2rgb(batch[63]).astype(np.uint8))  # Convert to uint8 to get correct colours
        plt.show()
