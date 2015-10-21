To run the Lenet convolutional neural network on the GPU use the command below:
```
$] THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python lenet.py
```