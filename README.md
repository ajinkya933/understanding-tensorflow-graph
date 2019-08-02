# understanding-tensorflow-graph
The code has been open source on github, https://github.com/fengjian0106/hed-tutorial-for-document-scanning
Ive translated this article into english from this chinese blog: http://fengjian0106.github.io/2017/05/08/Document-Scanning-With-TensorFlow-And-OpenCV/

## Foreword:
- This article is not an introductory tutorial for neural networks or machine learning. Instead, it demonstrates 
a key technical point for running a neural network on a mobile client through a real product case.
- In the field of convolutional neural networks, some very classic image classification networks have emerged, such as VGG16/VGG19, Inception v1-v4 Net, ResNet, etc. These classification networks can usually be used as the basic network structure in other algorithms. , especially the VGG network, is borrowed by many other algorithms. This article will also use the basic network structure of VGG16, but will not do a detailed introduction to the VGG network.
- Although this article is not an introductory tutorial on neural networking techniques, it will still provide a series of links to relevant introductory tutorials and technical documentation to help you understand the content of this article.
- The specific neural network algorithm used is only one component of this paper. In addition, this paper also introduces how to tailor the TensorFlow static library to run on the mobile phone, how to prepare training sample pictures, and various kinds of training neural networks. Skills, etc.

## Use Case

![alt text](https://github.com/ajinkya933/understanding-tensorflow-graph/blob/master/image1.png)

- The requirements are easy to describe clearly, as shown above, in a picture, the coordinates of the four vertices of a rectangular-shaped document are found.

## Traditional technical solutions

Google search opencv scan document, you can find several related tutorials, the technical means in these tutorials are similar, the key step is to call two functions in OpenCV, cv2.Canny () and cv2.findContours (). It seems easy to implement, but the real situation is that these tutorials are just a demo demo. The images used for the demonstrations are the most ideal simple cases. The real scene images will be more complicated than this. Various interference factors, the edge detection result obtained by calling the canny function will be more messy than the situation in the demo. For example, many lines of various lengths will be detected, or the edge line of the document will be cut into several short ones. Line segments, there are also gaps between the line segments with different distances. In addition, the findContours function can only detect the vertices of a closed polygon, but it does not ensure that the polygon is a reasonable rectangle. So in our first version of the technical solution, a lot of improvements and tunings were made to these two key steps, which are summarized as follows:

- Improve the effect of the canny algorithm, add extra steps, and get a better edge detection map.
- For the edge map obtained by the canny step, a set of mathematical algorithms is established to find a reasonable rectangular region from the edge map.

## Effective neural network algorithm

Several neural network algorithms tried in the past can't get the desired effect. Later, I changed the idea. Since the traditional technical means contains two key steps, can we use neural network to improve these two separately? Steps, after analysis, we can try to replace the canny algorithm with neural network, that is, use neural network to detect the edge of the rectangular region in the image. As long as this edge detection can remove more interference factors, the second step The algorithm inside can be made simpler.

## Neural network input and output
![alt text](https://github.com/ajinkya933/understanding-tensorflow-graph/blob/master/2.png)

According to this idea, for the neural network part, the current demand becomes the same as shown in the above figure.

## HED (Holistically-Nested Edge Detection) network

The need for edge detection, in the field of image processing, commonly called Edge Detection or Contour Detection, follows this idea and finds the Holistically-Nested Edge Detection network model.

The HED network model is designed based on the VGG16 network structure, so it is necessary to look at VGG16 first.

![alt text](https://github.com/ajinkya933/understanding-tensorflow-graph/blob/master/3.png)

The figure above is the schematic diagram of VGG16. In order to facilitate the transition from VGG16 to HED, we first turn VGG16 into the following diagram:

![alt text](https://github.com/ajinkya933/understanding-tensorflow-graph/blob/master/4.png)

In the above diagram, the different components of VGG16 are distinguished by different colors.

![alt text](https://github.com/ajinkya933/understanding-tensorflow-graph/blob/master/5.png)

As can be seen from the schematic diagram, the convolutional layer represented by green and the pooled layer represented by red can be clearly divided into five groups. The third group is shown in the above figure with a purple line.

![alt text](https://github.com/ajinkya933/understanding-tensorflow-graph/blob/master/6.png)

The HED network uses the five groups in the VGG16 network. The fully connected layer and the softmax layer in the latter part are not required. In addition, the pooling layer (red) of the fifth group is not required.

![alt text](https://github.com/ajinkya933/understanding-tensorflow-graph/blob/master/7.png)

After removing the unnecessary parts, you get the network structure like the above picture. Because of the role of the pooling layer, starting from the second group, the length and width of the input image of each group are the input images of the previous group. Half the length and width.

![alt text](https://github.com/ajinkya933/understanding-tensorflow-graph/blob/master/8.png)

The HED network is a multi-scale and multi-level feature learning network structure. The so-called multi-scale is the last convolution layer (green part) of each group of VGG16 as shown in the above figure. The output of the image is taken out, because the length and width of the image obtained by each group are different, so you need to use transposed convolution/deconvolution (deconv) for each group of images. Doing a calculation, in effect, is equivalent to expanding the length and width of the image obtained in the second to fifth groups by 2 to 16 times, so that the image obtained on each scale (each group of VGG16 is a scale) They are all the same size.

![alt text](https://github.com/ajinkya933/understanding-tensorflow-graph/blob/master/9.png)

Combine the same size images obtained on each scale, and get the final output image, which is the image with edge detection.

## The HED network structure code written based on TensorFlow is as follows:
```
def hed_net(inputs, batch_size):
    # ref https://github.com/s9xie/hed/blob/master/examples/hed/train_val.prototxt
    with tf.variable_scope('hed', 'hed', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
            # vgg16 conv && max_pool layers
            net = slim.repeat(inputs, 2, slim.conv2d, 12, [3, 3], scope='conv1')
            dsn1 = net
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            net = slim.repeat(net, 2, slim.conv2d, 24, [3, 3], scope='conv2')
            dsn2 = net
            net = slim.max_pool2d(net, [2, 2], scope='pool2')

            net = slim.repeat(net, 3, slim.conv2d, 48, [3, 3], scope='conv3')
            dsn3 = net
            net = slim.max_pool2d(net, [2, 2], scope='pool3')

            net = slim.repeat(net, 3, slim.conv2d, 96, [3, 3], scope='conv4')
            dsn4 = net
            net = slim.max_pool2d(net, [2, 2], scope='pool4')

            net = slim.repeat(net, 3, slim.conv2d, 192, [3, 3], scope='conv5')
            dsn5 = net
            # net = slim.max_pool2d(net, [2, 2], scope='pool5') # no need this pool layer

            # dsn layers
            dsn1 = slim.conv2d(dsn1, 1, [1, 1], scope='dsn1')
            # no need deconv for dsn1

            dsn2 = slim.conv2d(dsn2, 1, [1, 1], scope='dsn2')
            deconv_shape = tf.pack([batch_size, const.image_height, const.image_width, 1])
            dsn2 = deconv_mobile_version(dsn2, 2, deconv_shape) # deconv_mobile_version can work on mobile

            dsn3 = slim.conv2d(dsn3, 1, [1, 1], scope='dsn3')
            deconv_shape = tf.pack([batch_size, const.image_height, const.image_width, 1])
            dsn3 = deconv_mobile_version(dsn3, 4, deconv_shape)

            dsn4 = slim.conv2d(dsn4, 1, [1, 1], scope='dsn4')
            deconv_shape = tf.pack([batch_size, const.image_height, const.image_width, 1])
            dsn4 = deconv_mobile_version(dsn4, 8, deconv_shape)

            dsn5 = slim.conv2d(dsn5, 1, [1, 1], scope='dsn5')
            deconv_shape = tf.pack([batch_size, const.image_height, const.image_width, 1])
            dsn5 = deconv_mobile_version(dsn5, 16, deconv_shape)

            # dsn fuse
            dsn_fuse = tf.concat(3, [dsn1, dsn2, dsn3, dsn4, dsn5])
            dsn_fuse = tf.reshape(dsn_fuse, [batch_size, const.image_height, const.image_width, 5]) #without this, will get error: ValueError: Number of in_channels must be known.

            dsn_fuse = slim.conv2d(dsn_fuse, 1, [1, 1], scope='dsn_fuse')

    return dsn_fuse, dsn1, dsn2, dsn3, dsn4, dsn5

```

### Training network
## Cost function

The HED network given in the paper is a general edge detection network. According to the description of the paper, the image obtained at each scale needs to participate in the calculation of cost. The code of this part is as follows:

```
input_queue_for_train = tf.train.string_input_producer([FLAGS.csv_path])
image_tensor, annotation_tensor = input_image_pipeline(dataset_root_dir_string, input_queue_for_train, FLAGS.batch_size)

dsn_fuse, dsn1, dsn2, dsn3, dsn4, dsn5 = hed_net(image_tensor, FLAGS.batch_size)

cost = class_balanced_sigmoid_cross_entropy(dsn_fuse, annotation_tensor) + \
       class_balanced_sigmoid_cross_entropy(dsn1, annotation_tensor) + \
       class_balanced_sigmoid_cross_entropy(dsn2, annotation_tensor) + \
       class_balanced_sigmoid_cross_entropy(dsn3, annotation_tensor) + \
       class_balanced_sigmoid_cross_entropy(dsn4, annotation_tensor) + \
       class_balanced_sigmoid_cross_entropy(dsn5, annotation_tensor)
  ```
  
  In the network trained in this way, the detected edge line is a bit thick. In order to get a finer edge line, an optimization scheme is found through multiple experiments. The code is as follows:
  
  ```
input_queue_for_train = tf.train.string_input_producer([FLAGS.csv_path])
image_tensor, annotation_tensor = input_image_pipeline(dataset_root_dir_string, input_queue_for_train, FLAGS.batch_size)

dsn_fuse, _, _, _, _, _ = hed_net(image_tensor, FLAGS.batch_size)

cost = class_balanced_sigmoid_cross_entropy(dsn_fuse, annotation_tensor)
  ```
That is to say, the image obtained at each scale is no longer involved in the calculation of cost, and only the final image obtained after fusion is used for calculation.


### The effect of the two cost functions is shown in the following figure. The right side is the optimized effect:

![alt text](https://github.com/ajinkya933/understanding-tensorflow-graph/blob/master/10.png)

In addition, according to the requirements in the HED paper, when calculating the cost, you can not use the common variance cost, but should use the cost-sensitive loss function, the code is as follows:

```
def class_balanced_sigmoid_cross_entropy(logits, label, name='cross_entropy_loss'):
    """
    The class-balanced cross entropy loss,
    as in `Holistically-Nested Edge Detection
    <http://arxiv.org/abs/1504.06375>`_.
    This is more numerically stable than class_balanced_cross_entropy

    :param logits: size: the logits.
    :param label: size: the ground truth in {0,1}, of the same shape as logits.
    :returns: a scalar. class-balanced cross entropy loss
    """
    y = tf.cast(label, tf.float32)

    count_neg = tf.reduce_sum(1. - y) # the number of 0 in y
    count_pos = tf.reduce_sum(y) # the number of 1 in y (less than count_neg)
    beta = count_neg / (count_neg + count_pos)

    pos_weight = beta / (1 - beta)
    cost = tf.nn.weighted_cross_entropy_with_logits(logits, y, pos_weight)
    cost = tf.reduce_mean(cost * (1 - beta), name=name)

    return cost
```

## Bilinear initialization of transposed convolutional layers

When trying the FCN network, it was stuck for a long time. According to the FCN requirements, when using transposed convolution/deconvolution (deconv), the convolution kernel should be used. The value is initialized to a bilinear upsampling kernel instead of the usual normal distribution random initialization, while using a small learning rate, which makes it easier to converge the model.

In HED's paper, there is no explicit requirement to initialize the transposed convolution layer in this way. However, during the training process, it is found that the model is easier to converge when initialized in this way.

The code for this part is as follows:

```
def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    filter_size = get_kernel_size(factor)
    
    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)
    
    upsample_kernel = upsample_filt(filter_size)
    
    for i in xrange(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    
    return weights


```
## Cold start of training process

The HED network does not enter the convergence state as easily as the VGG network, and it is not easy to enter the desired ideal state, mainly for two reasons:

    The bilinear initialization of the transposed convolutional layer mentioned above is an important factor because deconvolution is required on all four scales. If the deconvolution layer cannot converge, the entire HED will not enter the desired Ideal state
    Another reason is caused by the multi-scale of HED. Since it is multi-scale, the image obtained on each scale should contribute to the final output image of the model. During the training process, if the size of the input image is It is 224*224, it is very easy to train successfully, but when the size of the input image is adjusted to 256*256, it is easy to have a situation, that is, the image obtained at 5 scales, there will be 1 ~ 2 images Is invalid (all black)

In order to solve the problem encountered here, the method adopted is to train the network with a small number of sample pictures (such as 2000 sheets). In a short training time (such as 1000 iterations), if the HED network cannot show a convergence trend, or If you can't reach the full effective state of the image at 5 scales, then simply give up the training results of this round, restart the next round of training until you are satisfied, and then continue to train the network using the complete training sample set.

### Training data set (large amount of synthetic data + small amount of real data)
The training dataset used in the HED paper is for general edge detection purposes, and what shapes have edges, such as the following:

![alt text](https://github.com/ajinkya933/understanding-tensorflow-graph/blob/master/11)

The model trained with this data, when doing document scanning, the detected edge effect is not ideal, and the sample size of this training data set is also very small, only a hundred pictures (because of this picture Manual labeling costs are too high), which also affects the quality of the model.

In the current demand, it is necessary to detect a rectangular area with a certain perspective and rotation transformation effect, so it is possible to boldly guess that if a more targeted training sample is prepared, it is possible to obtain a better edge detection effect.

With the real scene image collected from the first version of the technical solution, we developed a simple set of annotation tools, manually labeled 1200 images (the time cost of marking 1200 images is also very high), but these more than 1200 images are still There are a lot of problems. For example, for the neural network, 1200 training samples are actually not enough. In addition, the scenes covered by these pictures are actually few, and the similarity of some pictures is relatively high. Such data is put into the neural network for training. The effect of generalization is not good.


Therefore, more than 80,000 training sample images were synthesized using technical means.

![alt text](https://github.com/ajinkya933/understanding-tensorflow-graph/blob/master/13.png)

As shown in the figure above, a background image and a foreground image can be combined to form a pair of training sample data. In the process of synthesizing pictures, the following techniques and techniques are used:

- Add rotation, translation, perspective transformation on the foreground image
- Randomly cropped the background image
- Generate experimentally contrasted edges to create edge lines of appropriate width
- OpenCV does not support rotation and perspective transformation between transparent layers. Only the lowest precision interpolation algorithm can be used. In order to improve this, it is later changed to use iOS simulator to synthesize images through operations on CALayer.
- In the process of continuously improving the training samples, according to the statistics of the real sample pictures and the feedback information of various ways, some more complicated sample scenes are deliberately simulated, such as a messy background environment, linear edge interference, etc.

After continuous adjustment and optimization, we finally train a satisfactory model. We can see the edge detection effect of the neural network model again through the second column in the following chart:

![alt text](https://github.com/ajinkya933/understanding-tensorflow-graph/blob/master/14.png)

## Run TensorFlow on your mobile device
### Use the TensorFlow library on your phone

TensorFlow officially supports iOS and Android, and has clear documentation, just do it. But because TensorFlow is dependent on protobuf 3, there are some other problems that may be encountered, such as the following two, which are the problems and solutions we encountered in two different iOS apps, which can be used as a reference:

- A product uses protobuf 2, and for various historical reasons, it uses and stays on a very old version of the Base library, and the internal library is also used in protobuf 3, when the A product is upgraded to protobuf 3 The base library of protobuf 3 and the base library in the A source code generate some strange conflicts. The final solution is to manually modify the Base library in the A source code to avoid compile conflicts.

- The B product is also used by protobuf 2, and the multiple third-party modules used by the B product (no source code, only binary files) are also dependent on protobuf 2. It is not feasible to directly upgrade the protobuf library used by the B product. The last method is Modify the source code of protobuf 3 used in TensorFlow and TensorFlow, and replace protobuf 3 with a namespace so that two different versions of protobuf libraries can coexist.

Android can use dynamic libraries itself, so even if the app must use protobuf 2, it is okay for different modules to use dlopen to load the specific version of the library they need.

## Use trained model files on your phone
Models are usually trained on the PC side. For most users, the code is written in Python to get the model file in ckpt format. When using a model file, one way is to rebuild the complete neural network with the code, and then load the model file in ckpt format. If you use the model file on the PC, this method is actually acceptable, copy and paste. You can rebuild the entire neural network with Python code. However, you can only use the C++ interface provided by TensorFlow on your mobile phone. If you still use the same idea, you need to rebuild the neural network with C++ API. The workload is a bit big, and the C++ API is more complicated than the Python API. Many, so after training the network on the PC, you also need to convert the model file in ckpt format into a model file in pb format. The model file in pb format is a binary file serialized with protobuf, which contains the neural network. The specific structure and the value of each matrix, when using this pb file, you do not need to use code to build a complete neural network structure, just need to deserialize it, then the code written in C++ API will It's a lot simpler. In fact, this is also the recommended method of using TensorFlow. When using the model on a PC, you should also use this pb file (using the ckpt file during training).

## Strange crash encountered on the HED network on the phone
When loading the pb model file on the phone and running it, I encountered a weird error, as follows:

```
Invalid argument: No OpKernel was registered to support Op 'Mul' with these attrs.  Registered devices: [CPU], Registered kernels:
  device='CPU'; T in [DT_FLOAT]

	 [[Node: hed/mul_1 = Mul[T=DT_INT32](hed/strided_slice_2, hed/mul_1/y)]]
```

The reason is different because literally, the meaning of this error is the lack of multiplication (Mul), but I have compared it with other neural network models, the multiplication operation module can work normally.

After Google search, many people have encountered similar situations, but the error messages are different. Later, in Gensub issues of TensorFlow, they finally found clues, which are explained together because TensorFlow is modularized based on operation. And coding, each mathematical calculation module is an Operation, for various reasons, such as memory footprint, GPU exclusive operation, etc., the mobile version of TensorFlow does not contain all the Operation, the mobile version of TensorFlow supports the Operation only PC A subset of the full version of TensorFlow, I encountered this error because one of the operations used does not support the mobile version.

According to this clue, the Python code is checked one by one, and then the problematic code is located. The code before and after the modification is as follows:

```
def deconv(inputs, upsample_factor):
    input_shape = tf.shape(inputs)

    # Calculate the ouput size of the upsampled tensor
    upsampled_shape = tf.pack([input_shape[0],
                               input_shape[1] * upsample_factor,
                               input_shape[2] * upsample_factor,
                               1])

    upsample_filter_np = bilinear_upsample_weights(upsample_factor, 1)
    upsample_filter_tensor = tf.constant(upsample_filter_np)

    # Perform the upsampling
    upsampled_inputs = tf.nn.conv2d_transpose(inputs, upsample_filter_tensor,
                                              output_shape=upsampled_shape,
                                              strides=[1, upsample_factor, upsample_factor, 1])

    return upsampled_inputs

def deconv_mobile_version(inputs, upsample_factor, upsampled_shape):
    upsample_filter_np = bilinear_upsample_weights(upsample_factor, 1)
    upsample_filter_tensor = tf.constant(upsample_filter_np)

    # Perform the upsampling
    upsampled_inputs = tf.nn.conv2d_transpose(inputs, upsample_filter_tensor,
                                              output_shape=upsampled_shape,
                                              strides=[1, upsample_factor, upsample_factor, 1])

    return upsampled_inputs

```

The problem is caused by the two operations tf.shape and tf.pack in the deconv function. In the PC version code, for the sake of brevity, based on these two operations, the upsampled_shape is automatically calculated. After the modification, the caller is required. Set the corresponding upsampled_shape by hard coding.

## Crop TensorFlow

TensorFlow is a very large framework. For mobile phones, it takes up a lot of space, so you need to minimize the size of the TensorFlow library.

In fact, when solving the crash problem encountered earlier, I have already pointed out a tailoring idea. Since the mobile version of TensorFlow is originally a subset of the PC version, it means that this subset can be made according to specific needs. It becomes smaller, which also achieves the purpose of cutting. Specifically, modify the tensorflow/tensorflow/contrib/makefile/tf_op_files.txt file in the TensorFlow source code to keep only the modules that are used. For the HED network, the original 200+ modules were cropped to only 46, and the cropped tf_op_files.txt file is as follows:

```
tensorflow/core/kernels/xent_op.cc
tensorflow/core/kernels/where_op.cc
tensorflow/core/kernels/unpack_op.cc
tensorflow/core/kernels/transpose_op.cc
tensorflow/core/kernels/transpose_functor_cpu.cc
tensorflow/core/kernels/tensor_array_ops.cc
tensorflow/core/kernels/tensor_array.cc
tensorflow/core/kernels/split_op.cc
tensorflow/core/kernels/split_v_op.cc
tensorflow/core/kernels/split_lib_cpu.cc
tensorflow/core/kernels/shape_ops.cc
tensorflow/core/kernels/session_ops.cc
tensorflow/core/kernels/sendrecv_ops.cc
tensorflow/core/kernels/reverse_op.cc
tensorflow/core/kernels/reshape_op.cc
tensorflow/core/kernels/relu_op.cc
tensorflow/core/kernels/pooling_ops_common.cc
tensorflow/core/kernels/pack_op.cc
tensorflow/core/kernels/ops_util.cc
tensorflow/core/kernels/no_op.cc
tensorflow/core/kernels/maxpooling_op.cc
tensorflow/core/kernels/matmul_op.cc
tensorflow/core/kernels/immutable_constant_op.cc
tensorflow/core/kernels/identity_op.cc
tensorflow/core/kernels/gather_op.cc
tensorflow/core/kernels/gather_functor.cc
tensorflow/core/kernels/fill_functor.cc
tensorflow/core/kernels/dense_update_ops.cc
tensorflow/core/kernels/deep_conv2d.cc
tensorflow/core/kernels/xsmm_conv2d.cc
tensorflow/core/kernels/conv_ops_using_gemm.cc
tensorflow/core/kernels/conv_ops_fused.cc
tensorflow/core/kernels/conv_ops.cc
tensorflow/core/kernels/conv_grad_filter_ops.cc
tensorflow/core/kernels/conv_grad_input_ops.cc
tensorflow/core/kernels/conv_grad_ops.cc
tensorflow/core/kernels/constant_op.cc
tensorflow/core/kernels/concat_op.cc
tensorflow/core/kernels/concat_lib_cpu.cc
tensorflow/core/kernels/bias_op.cc
tensorflow/core/ops/sendrecv_ops.cc
tensorflow/core/ops/no_op.cc
tensorflow/core/ops/nn_ops.cc
tensorflow/core/ops/nn_grad.cc
tensorflow/core/ops/array_ops.cc
tensorflow/core/ops/array_grad.cc

```
One point to emphasize is that this kind of operation idea is different for different neural network structures. The principle is to reserve what module to use. Of course, because there are still hidden dependencies between some modules, it is necessary to repeatedly try repeatedly to be successful.

In addition, the following general methods can also achieve the purpose of cutting:

- Compiler-level strip operations automatically remove functions that are not called when linking (these development parameters usually have these parameters automatically set to the optimal combination)

- Slimming binary files with advanced techniques and tools

With all these cutting methods, the size of our ipa package was only increased by 3M. If you do not do manual cropping, then the increment of ipa is about 30M.

## Crop HED network

According to the reference information given in the HED paper, the size of the obtained model file is 56M, which is relatively large for mobile phones, and the larger the model, the larger the calculation amount, so it is necessary to consider whether the HED network can also be tailored. .

The HED network uses VGG16 as the underlying network structure, and VGG is a well-proven basic network structure. Therefore, modifying the overall structure of HED is certainly not a wise choice, at least not the preferred solution.

Considering the current demand, only detecting the edge of the rectangular region, instead of detecting the generalized edge in the general scene, it can be considered that the complexity of the former is lower than the latter, so a feasible idea is to preserve the overall structure of the HED. Modify the number of convolution kernels in each set of VGG convolutional layers to make the HED network more "thin".

According to this idea, after many adjustments and attempts, the number of parameters of a suitable convolution kernel is finally obtained. The corresponding model file is only 4.2M. On the iPhone 7P, the time consumption for processing each frame is about 0.1 second. , to meet the requirements of real-time.

The tailoring of neural networks is currently a hot topic in academia. There are several different theories to achieve tailoring for different purposes. However, it does not mean that every network structure has a clipping space, usually Should be combined with the actual situation, using the appropriate technical means, choose a model file of the appropriate size.

## Choice of TensorFlow API

TensorFlow's API is very flexible and relatively low-level. During the learning process, the code written by everyone varies greatly in style, and many engineers use various techniques to simplify the code, but this is actually the opposite. Invisibly increases the difficulty of reading the code, and is not conducive to the reuse of code.

Third-party communities and TensorFlow officials are aware of this problem, so it's better to use a more packaged, yet flexible API for development. The code in this article was written using TensorFlow-Slim.

## OpenCV algorithm
Although neural network technology has obtained a better edge detection effect than the canny algorithm, the neural network is not omnipotent, and the interference still exists. Therefore, the mathematical model algorithm in the second step is still What is needed is that because the edge detection in the first step has been greatly improved, the algorithm in the second step is appropriately simplified, and the overall adaptability of the algorithm is also stronger.

The algorithm in this part is shown below:
![alt text](https://github.com/ajinkya933/understanding-tensorflow-graph/blob/master/15.png)

In numerical order, several key steps do the following:

Using the HED network to detect the edges, you can see that the edge lines obtained here still have some interference.
On the image obtained in the previous step, use the HoughLinesP function to detect the line segment (blue line segment)
Extend the line segment obtained in the previous step to a straight line (green straight line)
Some of the line segments detected in the second step are very close, or some short line segments can be connected into a longer line segment, so some strategies can be used to merge them together. The straight line obtained in three steps. Define a strategy to determine whether two lines are equal. When two equal lines are encountered, the corresponding line segments of the two lines are merged or joined into one line segment. After this step is completed, the next step requires only the blue line segment without the green line.

According to the line segments obtained in the fourth step, calculate the intersection between them, and the adjacent intersections can also be merged. At the same time, each intersection and the line segment that produces the intersection are also associated (each blue point) , there is a set of red line segments associated with it)
For all the intersections obtained in the fifth step, take out 4 of them at a time, and judge whether the quadrilateral composed of the four points is a reasonable rectangle (a rectangle with a perspective transformation effect), except for conventional judgment strategies, such as angle, In addition to the ratio of the side lengths, there is also a judgment condition as to whether each side can coincide with the associated line segment of the corresponding point obtained in the fifth step. If it cannot be overlapped, the quadrilateral is unlikely to be detected by us. rectangle
After the sixth step of filtering, if you get multiple quads, you can use a simple filtering strategy, such as sorting to find the rectangle with the largest circumference or area.

Engineering angle

    When the end-to-end network is invalid, you can use the pipeline idea to consider the problem, split the service, and use the neural network technology in a targeted manner.
    At least master the development framework of a neural network, and pursue the engineering quality of the code.
    To master some basic routines in neural network technology,
    To find a balance between academia and industry, learn as many neural network models as possible in different problem areas as a technical reserve.
    
References:

    
