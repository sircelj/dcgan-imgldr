import unittest
from image_loader import SCC
from scipy.io.wavfile import read as wavread
import numpy as np
import copy


class ImageTest(unittest.TestCase):

    def test_inversion_correct(self):
        scc = SCC("../test/sound/", 4, sub_dirs=["no/", "off/", "on/", "yes/"])
        file_names = copy.copy(scc.file_names)
        self.assertTrue(file_names, "No files found.")
        images = scc.get_new_batch()

        for mag_phase, name in zip(images, file_names):
            audio = scc._image_to_audio(mag_phase)
            rate_orig, audio_orig = wavread(scc.im_dir + name)
            rate_orig, audio_orig = scc._downsample(rate_orig, audio_orig)
            np.testing.assert_array_almost_equal(audio[:len(audio_orig)], audio_orig, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
