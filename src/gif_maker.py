import imageio
import os


# CelebA generated images
dir1 = "celeba_output/2018-07-30_22h50m26s_DCGAN_S/images/"
images = []
for i, file_name in enumerate(os.listdir(dir1)):
    print(file_name)
    image = imageio.imread(dir1 + file_name)
    image[0:13, 0:int(646 * (i) / 14)] = 0
    images.append(image)
imageio.mimsave("celeba_output/2018-07-30_22h50m26s_DCGAN_S/images.gif", images, duration=0.5)


# SCC generated mag/phase plots
dir2 =  "celeba_output/2018-09-11_18h28m14s_DCGAN_12_SCC/images/"
images = []
i = 0
for dir_name in os.listdir(dir2):
    if os.path.isdir(dir2 + dir_name):
        print(dir2 + dir_name + "/mag_phase.png")
        image = imageio.imread(dir2 + dir_name + "/mag_phase.png")
        image[0:13, 0:int(646 * i / 87)] = 0
        images.append(image)
    i += 1
imageio.mimsave("celeba_output/2018-09-11_18h28m14s_DCGAN_12_SCC/images.gif", images, duration=0.5)
