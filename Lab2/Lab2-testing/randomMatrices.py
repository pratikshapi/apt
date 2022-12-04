import numpy as np
k=20
array = np.asarray(np.random.randint(2000, size=(k, k)))
np.set_printoptions(precision = 0, suppress = True)

with open(str(k)+'.txt','wb') as f:
    np.savetxt(f, array, delimiter=' ', fmt='%s')


# with open(str(k)+'.txt','wb') as f:
#     for line in array:
#         print(line)
        
#         np.tofile(f, sep=' ', line, format='%.2f')
# print(array) 