# understanding-tensorflow-graph
The code has been open source on github, https://github.com/fengjian0106/hed-tutorial-for-document-scanning

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



