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
