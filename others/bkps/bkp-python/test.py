# importing the modules
import numpy as np
import cv2

# creating array using np.zeroes()
array = np.zeros([16, 16, 3],
                 dtype = np.uint8)
  
# setting RGB color values as 255,255,255
array[:, :] = [255, 255, 255] 
  
# displaying the image
cv2.imwrite("image.bmp", array)






# from os import POSIX_FADV_RANDOM
# from PIL import Image # import library

# img = Image.new(mode = "RGB", size = (16,16)) # creates a RGB image with the size of 1x1
# pixels = img.load() # Creates the pixel map
# pixels(255,255,255) # Set the colour of the first (and only) pixel to white



# import numpy as np
# a_shape = [16, 16, 3]
# a = np.ones(a_shape, dtype = np.uint8)
# a[:, :] = [255, 255, 255] 
# print(a)
# img = Image.new(mode = "RGB", size = (16,16))
# pixels = img.load()
# pixels = a
# print('pixels')
# print(pixels)
# img.save(format='bmp', fp='./graphics/test.bmp')




# from PIL import Image
# import numpy as np
# from matplotlib import cm

# im = Image.open("graphics/pac-man.bmp")
# p = np.array(im)
# # print(p)
# # print(p.shape)
# # (728, 641, 3)
# # bmp_map = [[255 for i in range(3)] for j in range(641)] for k in range(728)]
# # 3 - powerup, 2 - coin, 1 - wall, 0 space

# bmp_map=np.full((728, 641), 0)
# print(bmp_map.shape)

# for ii, i in enumerate(p):
#     for jj, j in enumerate(i):
#         # print(j>=240)
#         # if (j>=240)
#         if np.count_nonzero(j<=10) == 3:
#             bmp_map[ii][jj] = 1
#         # for k in j:
#         #     pass


# print(bmp_map)

# print('out of loop?')
# with open('test.txt', 'wb') as f:
#     print('saving?')
#     np.save(f, bmp_map)

# # Image.fromarray(np.uint8(p))

# # im2 = Image.fromarray(np.uint8(cm.gist_earth(p[0])))
# # im2.save('test.bmp')