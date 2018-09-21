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
    """ImageLoader used for the CelebA dataset.

    Handles the process of reading images and returning batches of
    them. Internally also handles if an epoch is finished,
    randomizing the dataset.

    Parameters
    ----------
    im_dir : string
        Relative or full directory name containing the images.
    batch_size : int
        Desired output batch size of the ImageLoader
    sub_dirs : array
        Array of subdirectories specified as strings. They must have
        a "/" delimiter at the end.
    start_epoch : int
        Epoch number the ImageLoader startss from. Can get overwritten
        by self.set_epoch()
    im_width : int
        Width of the cropped images.
    im_height : int
        Height of the cropped images.
    im_c : int
        Number of channels the images have.

    Attributes
    ----------
    im_dir : string
        Image set directory
    image_index : int
        Starting index of the next batch to ge returned.
    epoch : int
        Current epoch the ImageLoader is in.
    file_names : array
        Array of file names. The names are relative to the im_dir
    number_of_images : int
        Number of images in the set.

    """
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
        """
        Plot a 8 x 8 grid of images from samples and return the figure.
        You need to call plt.show() of plt.savefig() to obtain the
        combined image. Pixel values should be in the range [-1, 1]

        Parameters
        ----------
        samples : ndarray, shape (batch_size, height, width, #colors)
            Array of images you want to plot. batch_size needs to be
            at least 64 big. Only the first 64 images will be plotted

        Returns
        -------
        fig : matplotlib.pyplot.figure object

        """
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
        """
        Set the epoch number to the ImageLoader. Used when initializing or
        restarting the GAN training.

        Parameters
        ----------
        epoch : int
            Desired starting epoch number.

        """
        self.epoch = epoch

    def _get_image(self, name):
        """
        Internal method. Reads the desired CelebA image and crops it to a
        specified size.

        Parameters
        ----------
        name : string
            Name of the image to be read.

        Returns
        -------
        image : ndarray, shape (self.im_width, self.im_height, self.im_c)
           Image represented as an numpy array. Pixel values are moved to
           a [-1, 1] range.

        """
        im = imread(self.im_dir + name)
        im_h, im_w, im_c = im.shape
        im = imresize(im[(im_h // 2 - 64):(im_h // 2 + 64), (im_w // 2 - 64):(im_w // 2 + 64), :],
                      [self.im_width, self.im_height])
        return self.rgb2tanh(im)

    def get_new_batch(self):
        """
        Get a new batch of images with the previously specified batch_size.
        Internally checks if all images in the set have been used. If this
        is the case it randomizes the set and increments the epoch counter.

        Returns
        -------
        batch : ndarray
            Batch of images

        """
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
        """
        Makes a figure using self.plotimage and saves a png into the
        specified dir with the epoch number as its name.

        Parameters
        ----------
        samples : ndarray
            Array of images you want to save. Needs to hold at least 64
            images
        dir_name : string
            Name of the target directory
        epoch : int
            Specifies the name of the saved figure. It is advisable that
            it is max 3 digits long, since the name format is "%00%d".

        """
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


class SCC(ImageLoader):
    """An ImageLoader but repurposed for the SCC data-set.

    Works similarly to ImageLoader. But, since it is dealing with audio,
    it transforms each signal into a STFT and converts to a polar form.
    Consequentially we obtain images with two channels, one for the
    magnitude, which is logarithmically transformed, and the other for
    the phase, which is scaled from [-Pi, Pi] to [-1, 1].

    Parameters
    ----------
    -> See ImageLoader

    Attributes
    ----------
    rate : int
        Frequency of the audio signals.
    step : int
        The signal rate is downsampled by 2. From 16000Hz to 8000Hz.
    image_size : tuple, shape (2, )
        Size of the combined mag/phase image used for plotting.

    """

    def __init__(self, im_dir, batch_size, sub_dirs=[""], start_epoch=0, im_width=128, im_height=64, im_c=2):
        super().__init__(im_dir, batch_size, sub_dirs, start_epoch, im_width, im_height, im_c)
        self.rate = None
        self.step = 2
        self.image_size = (128, 128)

        # Set the rate
        self._get_image(random.choice(self.file_names))

    def _downsample(self, rate, signal):
        """
        Method for downsampling the signals. Downsampling is performed by
        taking every self.step sample from the signal.

        Parameters
        ----------
        rate : int
            Original rate of the signal.
        signal : ndarray
            Original audio signal.

        Returns
        -------
        rate : int
            New rate of the signal
        signal : ndarray
            Downsampled signal.

        """
        return rate/self.step, signal[::self.step]

    def _get_image(self, name, nperseg=126, noverlap=None, mag_scale=np.log10(2**15)):
        """
        From audio in the file name construct the magnitude/phase tensor.

        Parameters
        ----------
        name : string
            Name of the audio file.
        nperseg : int
            Size of each FFT window for the STFT.
        noverlap : int or None
            Size of the overlap to the STFT. If None, then a half-step is
            used.
        mag_scale : float
            Value with which the magnitude will be scaled.

        Returns
        -------
        mag_phase : ndarray, shape (stft_width, stft_heigth, 2)
            The magnitude/phase tensor.
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

        # Join into a two-channel tensor
        return np.stack((phasegram, spectrogram), axis=-1)

    def _image_to_audio(self, mag_phase, nperseg=126, noverlap=None, mag_scale=np.log10(2**15)):
        """
        Convert the magnitude/phase tensor into audio signal.

        Parameters
        ----------
        mag_phase : ndarray
            The magnitude/phase tensor.
        nperseg : int
            Size of each FFT window for the STFT.
        noverlap : int or None
            Size of the overlap to the STFT. If None, then a half-step is
            used.
        mag_scale : float
            Value with which the magnitude will be scaled.

        Returns
        -------
        sig : ndarray
            Time domain signal.
        """

        # Extract magnitude and phase matrix from the mag/phase matrix
        width, height, _ = mag_phase.shape
        # if height != nperseg // 2:
        #     raise ValueError("mag/phase matrix height not consistent width nperseg")
        phasegram = np.pi * mag_phase[:, :, 0]
        spectrogram = mag_scale * mag_phase[:, :, 1]

        # Transform the mag/phase matrix to a SFTF matrix
        stft = np.power(10, spectrogram) * np.exp(1j * phasegram)

        # Do an inverse of the SFTF using the LSEE-MSTFT method from Griffin-Lim paper
        times, audio_sig = signal.istft(stft, fs=self.rate, nperseg=nperseg, noverlap=noverlap)
        return audio_sig

    def epoch_save(self, samples, dir_name, epoch):
        """
        Makes a mag/phase figure using _get_image and plotimage from
        samples and saves it as a .png file. Samples are also transformed
        to audio signals and saved as a .wav file. All files are saved
        into a subdirectory with the epoch number as its name.

        Parameters
        ----------
        samples : ndarray
            Array of mag/phase tensors you want to save. Needs to hold at
            least 64 tensors.
        dir_name : string
            Name of the target directory.
        epoch : int
            Specifies the name of directory containing the saved figure
            and .wav ffiles . It is advisable that it is max 3 digits
            long, since the name format is "%00%d".

        """

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
            wavwrite(ep_dir_name + "%s.wav" % str(i).zfill(3), int(self.rate), audio/self.rate)
