# 2019.12.03 change
#import scipy.misc
import imageio


try:
    # 2019.12.03 change
#    img = scipy.misc.imread('./example-image.png', mode='RGB')
    img = imageio.imread('./example-image.png', pilmode='RGB')
except AttributeError:
    s = ("scipy.misc.imread requires Python's image library PIL"
         " You can satisfy this requirement by installing the"
         " userfriendly fork PILLOW via `pip install pillow`.")
    raise AttributeError(s)
    
    
print('Image shape:', img.shape)
print('Number of channels:', img.shape[2])
print('Image data type:', img.dtype)

print(img[100:102, 100:102, :])
