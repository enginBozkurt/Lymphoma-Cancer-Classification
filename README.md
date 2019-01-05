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

<h2>
  
```diff
+ Making a database
```
</h2>
  
![screenshot_10](https://user-images.githubusercontent.com/30608533/50726745-7326e580-1122-11e9-8349-21a672af4ba5.jpg)

<p5>
  
Regardless of the desired model type used for the classifier, deep learning (DL) is typically best performed using some type of a database backend. This database need not be sophisticated (e.g., the LMDB commonly used by Caffe), nor must it be a true database (e.g., here we’ll discuss using PyTables which has an HDF5 backend). The benefits of using a database are numerous, but we can briefly list some of them here:
- As individual files, extracted image “patches” can often number into the thousands or millions in the case of DL. The access time can be very slow as modern operating systems are not typically tuned for efficient access of large numbers of files. Trying to do something as simple as “ls” in a directory in Windows/Linux with millions of files can cause a notable lag.

- Improved speed as a result of both reading off of disk, o/s level caching, and improved compression

- Improved reproducibility, coming back to a project later which has a database allows for much more consistent retraining of the model, especially when other files may have been lost (e.g., lists of training and testing sets)

- Better protection against data leakage. Although not required, I prefer creating separate databases (in this case files), for training, validation, and testing purposes. This can greatly reduce human error in making sure that only training images are read during training time, and that validation/test images are completely held out.

</p5>

![screenshot_2](https://user-images.githubusercontent.com/30608533/50726860-d5ccb100-1123-11e9-91fd-a7e2c4ccd407.jpg)


<p6>

Note that this code chops images into overlapping tiles, at a user specified stride, which is very fast based on python views. If your experiment focuses on only smaller annotated pieces of the image, this code would need to be adjusted (e.g., ROIs of localized disease presentation).
  
</p6>


<p7>
  
That said, some specific items worth pointing out in the code:

- Images are stored as unsigned 8-bit integers, and thus their values range from [0,255]

- We assume that each image represents a unique patient and thus naively splits the images into training and validation batches. If this is not the case, it should be addressed there by assigning appropriate files to each of the phase dictionary items. Always remember, training and validation should take place at a patient level (e., a patient is either in the training set or the testing set, but never both)

- We use a modest compression level of 6, which some experiments have shown to be a nice trade-off between time and size. This is easily modifiable in the code by changed the value of “complevel”

- Images are stored in the database in [IMAGE, NROW,NCOL,NCHANNEL] format. This allows for easier retrieval later. Likewise, the chunk size is set to the tile size such that it is compressed independently of the other items in the database, allowing for rapid data access.

- The class is determined by looking at the filename for one of the 3 specified labels. In this case, each of the classes is in its own unique directory, with the correct associated class name.

- The labels, filenames, and total number of instances of each class are stored in the database for downstream reference (the latter to allow for class weighting).

</p7>

<h2>

```diff
+ Training a model
```

</h2>

<p8>
  
  Now that once we have the data ready, we’ll train the network. The Densenet architecture is provided by PyTorch in the torchvision package, in a very modular fashion. Thus the main components that we need to develop and discuss here is how to get our data in and out of the network.
</p8>

<p9>

One important practice which is commonly overlooked is to visually examine a sample of the input which will be going to the network, which we do in this cell:
</p9>

![screenshot_3](https://user-images.githubusercontent.com/30608533/50727772-c4d66c80-1130-11e9-8a39-e3905fb008ec.jpg)

<p11>
  
We can see that the augmentations have been done properly (image looks “sane”). We can also note how the color augmentation has drastically changed the appearance of the image (left) from H&E to a greenish color space, in hopes of greater generalizability later. Here I’m using the default values as an example, but tuning these values will likely improve results if they’re tailored towards to specific test set of interest
.</p11>

<p12>
  
Some notes:
- In this example, we’ve used a reduced version of Densenet for prototyping (as defined by the parameters in the first cell). For production usage, these values should be tuned for optimal performance. In the 5th cell, one can see the default parameters. For an explination of the parameters, please refer to the both the code and manuscript links for Densenet provided above.
-  This code is heavily reused for both training and valuation through the notion of “phases”, in particular the cell labeled 134 contains a bit of intertwined code which sets appropriate backend functionality (e.g., enabling/disabling gradient computation)
- Note that the validation is step is very expensive in time, and should be enabled with care.
</p12>

<p13>
  
The bottom of the notebook shows how to both visualize individual kernels and to visualize activations. Note that to be efficient pytorch does not keep activations in memory after the network is done computing. Thus it is impossible to retrieve them after the model does its prediction. As a result, we create a hook which saves the activations we’re interested at when the layer of interest is encountered.
</p13>

![screenshot_4](https://user-images.githubusercontent.com/30608533/50728012-ae321480-1134-11e9-8ace-d27fe71b5560.jpg)

<h2>

```diff
+ Visualizing results in the validation set
```

</h2>


<p14>

Since we’re consistently saving the best model as the classifier is training, we can interrogate the results on the validation set easily while the network itself is training. This is best done if 2 GPUs are available, so that the main GPU can continue training while the second GPU can generate the output. If the network is small enough and the memory of a singular GPU large enough, both processes can be done using the same GPU.

Some notes:
-  Augmentation in the related cell: in the case of using this for output generation, we want to use the original images since they will give a better sense of the expected output when used on the rest of the dataset, as a result, we disable all unnecessary augmentation. The only component that remains here is the randomcrop, to ensure that regardless of the size of the image in the database, we extract an appropriately sized patch

-  We can see the output has 2 components. The numerical components show the ground truth class versus the predicted class, as well as the raw deep learning output (i.e., pre argmax). Additionally, we can see the input image after and before augmentation.
- At the bottom of the output, we can see the confusion matrix for this particular subset, with the accuracy (here shown to be 100% on 4 examples).

</p14>

![screenshot_5](https://user-images.githubusercontent.com/30608533/50728359-6eb9f700-1139-11e9-83d3-35f20810dc44.jpg)


