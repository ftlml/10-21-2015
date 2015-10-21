from theano.tensor.signal import downsample

import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image
import theano
import theano.tensor as T

# Load the demo image.
bugatti = Image.open(open('Bugatti-VGT.jpg'))
bugatti = np.array(bugatti, dtype = 'float32') / 256
bugatti_p = bugatti.transpose(2, 0, 1)
# Create a theano computation graph to subsample (pool)
# the image.
X = T.tensor3(name = 'X')
op = downsample.max_pool_2d(input = X, ds = (4, 4),
                           ignore_border = True)
pool = theano.function([X], outputs = op)
# Generate the feature map by convolving the filter with the image.
feat_map = pool(bugatti_p)
feat_map_p = feat_map.transpose(1, 2, 0)
# Plot the original image.
plt.subplot(1, 2, 1); plt.axis('off'); plt.imshow(bugatti)
# Plot the convolved image.
plt.subplot(1, 2, 2); plt.axis('off'); plt.imshow(feat_map_p)
# Show the results on the screen.
plt.show()