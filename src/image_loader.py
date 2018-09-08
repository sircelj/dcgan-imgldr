import os
import numpy as np
from scipy.misc import imresize, imread
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from scipy import signal
import random


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
    def __init__(self, im_dir, batch_size, sub_dirs=[""], start_epoch=0, im_width=64, im_height=64, im_c=3):
        self.im_dir = im_dir
        self.image_index = 0
        self.epoch = start_epoch
        self.batch_size = batch_size
        self.file_names = [s_dir + name for s_dir in sub_dirs for name in os.listdir(im_dir + s_dir)
                           if os.path.isfile(im_dir + s_dir + name)]
        np.random.shuffle(self.file_names)
        self.number_of_images = len(self.file_names)
        self.im_width = im_width
        self.im_height = im_height
        self.im_c = im_c
        self.image_size = (64, 64, 3)

    def plotimage(self, samples):
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(8, 8)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples[:64]):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(self.tanh2rgb(sample.reshape(*self.image_size)).astype(np.uint8), cmap='Greys_r')

        return fig

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _get_image(self, name):
        im = imread(self.im_dir + name)
        im_h, im_w, im_c = im.shape
        im = imresize(im[(im_h // 2 - 64):(im_h // 2 + 64), (im_w // 2 - 64):(im_w // 2 + 64), :],
                      [self.im_width, self.im_height])
        return self.rgb2tanh(im)

    def get_new_batch(self):
        batch = np.zeros([self.batch_size, self.im_height, self.im_width, self.im_c])
        for i, name in enumerate(self.file_names[self.image_index:self.image_index+self.batch_size]):
            im = self._get_image(name)
            batch[i] = im
        self.image_index += self.batch_size

        if self.image_index + self.batch_size > self.number_of_images:
            self.epoch += 1
            self.image_index = 0
            np.random.shuffle(self.file_names)

        return batch

    def epoch_save(self, samples, dir_name, epoch):
        fig = self.plotimage(samples)
        plt.savefig(dir_name + '%s.png' % str(epoch).zfill(3), bbox_inches='tight')
        plt.close(fig)

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


class SCC(ImageLoader):

    def __init__(self, im_dir, batch_size, sub_dirs=[""], start_epoch=0, im_width=128, im_height=64, im_c=2):
        super().__init__(im_dir, batch_size, sub_dirs, start_epoch, im_width, im_height, im_c)
        self.rate = None
        self.step = 2
        self.image_size = (128, 128)

        # Set the rate
        self._get_image(self.im_dir + random.choice(self.file_names))

    def _downsample(self, rate, signal):
        return rate/self.step, signal[::self.step]

    def _get_image(self, name, nperseg=126, noverlap=None, mag_scale=np.log10(2**15)):
        """
        From audio in name construct the magnitude/phase tensor.
        :param name: Name of the audio ffile
        :return: magnitude/phase matrix
        """

        # Read the audio and downscale the rate by 2
        rate, audio_sig = wavread(self.im_dir + name)
        rate, audio_sig = self._downsample(rate, audio_sig)

        # Set global rate
        if self.rate is None:
            self.rate = rate

        # Right pad audio to desired size
        if noverlap is None:
            noverlap = (nperseg + 1) // 2

        length_orig = len(audio_sig)
        length_pad = int(np.ceil(length_orig/ noverlap) * noverlap)
        audio_sig = np.pad(audio_sig, (0, length_pad - length_orig), 'constant')

        # Make a Short time Fourier transform
        frequencies, times, stft = signal.stft(audio_sig, fs=rate,
                                               nperseg=nperseg, noverlap=noverlap)

        # Convert to log10 magnitude and phase
        spectrogram = np.log10(np.absolute(stft) + 1e-10)
        phasegram = np.angle(stft) / np.pi  # Scale angles to [-1, 1]

        # Scale the magnitude
        spectrogram /= mag_scale

        if stft.shape[1] != 128:
            # Pad the matrices
            spectrogram = np.pad(spectrogram, [(0, 0), (0, 128 - stft.shape[1])], 'minimum')
            phasegram = np.pad(phasegram, [(0, 0), (0, 128 - stft.shape[1])], 'constant')

        # Join into one two channel tensor
        return np.stack((phasegram, spectrogram), axis=-1)

    def _image_to_audio(self, mag_phase, nperseg=126, noverlap=None, mag_scale=np.log10(2**15)):
        """
        Convert the magnitude/phase tensor into audio signal.
        :param mag_phase: Magnitude/phase tensor
        :return: Time domain signal
        """
        # Extract magnitude and phase matrix from the mag/phase matrix
        width, height, _ = mag_phase.shape
        # if height != nperseg // 2:
        #     raise ValueError("mag/phase matrix height not consistent width nperseg")
        phasegram = np.pi * mag_phase[:, :, 0]
        spectrogram = mag_scale * mag_phase[:, :, 1]

        # Todo: brisi to
        if False:
            plt.figure(1)
            plt.imshow(spectrogram)
            plt.figure(2)
            plt.imshow(phasegram)
            plt.show()

        # Transform the mag/phase matrix to a SFTF matrix
        stft = np.power(10, spectrogram) * np.exp(1j * phasegram)

        # Do an inverse of the SFTF using the LSEE-MSTFT method from Griffin-Lim paper
        times, audio_sig = signal.istft(stft, fs=self.rate, nperseg=nperseg, noverlap=noverlap)
        return audio_sig

    def epoch_save(self, samples, dir_name, epoch):

        # Create new epoch directory
        ep_dir_name = dir_name + '/%s' % str(epoch).zfill(3) + '/'
        if not os.path.exists(ep_dir_name):
            os.makedirs(ep_dir_name)

        # Create a 8x8 grid of mag/phase images
        batch_size, _, width, _ = samples.shape
        samples_magphase = np.zeros((batch_size, width, width))
        # samples_magphase[:, :width // 2, :] = np.squeeze(samples[:, :, :, 0])
        samples_magphase[:, :width // 2, :] = samples[:, :, :, 0]  # Spectrogram
        samples_magphase[:, width // 2:, :] = samples[:, :, :, 1]  # Phase

        # Save the image
        fig = self.plotimage(samples_magphase)
        plt.savefig(ep_dir_name + 'mag_phase.png', bbox_inches='tight')
        plt.close(fig)

        # Save sound samples
        for i, magphase in enumerate(samples):
            audio = self._image_to_audio(magphase)
            wavwrite(ep_dir_name + "%s.wav" % str(i).zfill(3), int(self.rate), audio)


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
