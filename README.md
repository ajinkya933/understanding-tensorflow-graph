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
