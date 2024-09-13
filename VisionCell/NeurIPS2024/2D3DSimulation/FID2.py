import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
from numpy.random import random
from PIL import Image

# calculate frechet inception distance
def calculate_fid(act1, act2):
    # reshape arrays to have single channel
    act1 = act1.reshape((act1.shape[0], -1))
    act2 = act2.reshape((act2.shape[0], -1))
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# load images
file1 = input("Enter file1 path: ")
file2 = input("Enter file2 path: ")
image1 = Image.open(file1).convert('RGB')
image2 = Image.open(file2).convert('RGB')
# convert images to numpy arrays
act1 = np.array(image1)
act2 = np.array(image2)

# fid between act1 and act1
fid = calculate_fid(act1, act1)
print('FID (same): %.3f' % fid)

# fid between act1 and act2
fid = calculate_fid(act1, act2)
print('FID (different): %.3f' % fid)
