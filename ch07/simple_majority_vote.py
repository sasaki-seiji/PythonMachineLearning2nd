import numpy as np

# 2020.01.06 change
#np.argmax(np.bincount([0, 0, 1], 
#                      weights=[0.2, 0.2, 0.6]))
p = np.argmax(np.bincount([0, 0, 1], 
                      weights=[0.2, 0.2, 0.6]))
print(p)



ex = np.array([[0.9, 0.1],
               [0.8, 0.2],
               [0.4, 0.6]])

p = np.average(ex, 
               axis=0, 
               weights=[0.2, 0.2, 0.6])
p
# 2020.01.06
print(p)




np.argmax(p)
