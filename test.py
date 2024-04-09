import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_path = '/kaggle/working/EfficientDet/datasets/varroa/test/2017-09-20_19-24-55-mp4-bee_id_4402-44400-1_png_jpg.rf.9da30e6bb3e44c6aa4a6281e4f0b2435.jpg'

img = mpimg.imread(image_path)
plt.imshow(img)
plt.axis('off')  # Hide axis to focus on the image
plt.show()
