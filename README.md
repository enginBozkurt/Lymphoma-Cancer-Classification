# Lymphoma Cancer Classification

## A PyTorch implementation for differentiating between different types of lymphoma cancer.

<h2>

```diff
+ General Introduction to DenseNet
```

</h2>

<p1>

Here I discuss only the high-level intuition of the DenseNets.  
DenseNets consist of multiple dense-blocks, which look like this:
</p1>

![1](https://user-images.githubusercontent.com/30608533/50726520-9ac87e80-111f-11e9-92b3-8184d09ad9ca.png)
<p2>

The SSD detector differs from others single shot detectors due to the usage of multiple layers that provide a finer accuracy on objects with different scales. (Each deeper layer will see bigger objects).
The SSD normally start with a VGG on Resnet pre-trained model that is converted to a fully convolution neural network. 
These blocks are the workhorse of the densenet. Inside this block, the output from each kernel is concatenated with all subsequent features. When looking at x_4, one can notice that there are 4 other additional inputs being fed into it (i.e., yellow, purple, green, and red), 1 from each of the previous convolutional layers. Similarly, x_3 has 3 inputs, x_2 has 2, and x_0 has none as it is the first convolutional layer.
Multiple sets of these blocks are then sequentially applied, with a bottleneck layer in between them to form the entire network, which looks like this:

</p2>

![densenet_figure2](https://user-images.githubusercontent.com/30608533/50726551-fb57bb80-111f-11e9-86ea-55e8044a7cd1.png)

<p3>

The authors of this approach claim “DenseNets exploit the potential of feature reuse, yielding condensed models that are easy to train and highly parameter efficient”.
Unpacking this, we can see the reasoning for these claims:
1) Directly connecting layers throughout the network helps to reduce the vanishing gradient problem
2) Features learned at the earlier layers, which likely contain important filters (for example edge detectors) can be reused in later networks directly as opposed to having to be relearned anew. This both (a) reduces the overall amount of feature redundancy (each layer doesn’t need to learn its own edge detector), resulting in fewer overall parameters, potentially less opportunities for overfitting, and (b) result in faster training times than e.g., ResNet (no additional computation required on “inherited” data in densenets, while resnets require additional operations)

</p3>

```diff
+ Making a database
```

![screenshot_10](https://user-images.githubusercontent.com/30608533/50726745-7326e580-1122-11e9-8349-21a672af4ba5.jpg)

