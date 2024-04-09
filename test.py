# Assuming the image has already been saved by your script:
image_path = '/kaggle/working/EfficientDet/test/inferred_2017-09-20_19-24-55-mp4-bee_id_4971-60225-1_png_jpg.rf.1715a8306f2c33b3d843e27f434be8fc.jpg'

# Using IPython.display to show the image
from IPython.display import Image, display
display(Image(filename=image_path))
