from image_loader import SCC
import matplotlib.pyplot as plt
import numpy as np


scc = SCC("../speech_small/", sub_dirs=["yes/", "no/", "on/", "off/"], batch_size=64)
batch = scc.get_new_batch()

scc.epoch_save(batch, "out_test/", 66)

# Create a 8x8 grid of mag/phase images
batch_size, _, width, _ = batch.shape
samples_magphase = np.zeros((batch_size, width, width))
samples_magphase[:, :width // 2, :] = batch[:, :, :, 0]  # Spectrogram
samples_magphase[:, width // 2:, :] = batch[:, :, :, 1]  # Phase

scc.plotimage(samples_magphase)
plt.savefig('scc_original.png', bbox_inches='tight')
