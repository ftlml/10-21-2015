import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image
import theano
import theano.tensor as T

# Load the demo image.
bugatti = Image.open(open('Bugatti-VGT.jpg'))
bugatti = np.array(bugatti, dtype = 'float32') / 256
bugatti_p = bugatti.transpose(2, 0, 1).reshape(1, 3, 524, 932)
# Create an edge detector filter.
filter = np.ndarray((1, 3, 3, 3), dtype = 'float32')
filter[:, :, :, :] = [[[[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]]]]
# Create a theano computation graph to convolve
# the image with the filter.
X = T.tensor4(name = 'X')
theta = T.tensor4(name = 'theta')
op = T.nnet.conv.conv2d(X, theta)
convolve = theano.function([X, theta], outputs = op)
# Generate the feature map by convolving the filter
# with the image.
feat_map = convolve(bugatti_p, filter)
# Plot the original (left) and convolved image (right).
plt.subplot(1, 2, 1); plt.axis('off'); plt.imshow(bugatti)
plt.gray()
plt.subplot(1, 2, 2); plt.axis('off'); plt.imshow(feat_map[0, 0, :, :])
# Show the results on the screen.
plt.show()