from image_loader import ImageLoader
import matplotlib.pyplot as plt


il = ImageLoader("../img_small/", 64)
batch = il.get_new_batch()
il.plotimage(batch)
plt.savefig('celeba_original.png', bbox_inches='tight')
