# PCA Demonstration

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import imageio
import numpy as np
from skimage import color

# dimensionality used 
numberComponents = 20

#read in and display the image
img = plt.imread('house.jpeg')
plt.imshow(img)
plt.show()

#convert the image to greyscale
img=color.rgb2gray(img)
plt.imshow(img, cmap='gray')
plt.show()

#apply PCA
pca = PCA(numberComponents)
img_transformed = pca.fit_transform(img)
img_inverted = pca.inverse_transform(img_transformed)

#display the result
plt.imshow(img_inverted,cmap='gray' )
plt.show()

# plot principal components themselves
comps=np.round(pca.explained_variance_ratio_*100, decimals = 2)
print(comps)










        



