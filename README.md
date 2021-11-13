# Gradient-weighted Class Activation Mapping (Grad-CAM)
## PyTorch example of an image classification problem using a Resnet-18 CNN
![alt text](https://github.com/nmaffe/Grad-CAM-ResnetCNN-ImageNet-PyTorch/blob/main/result/result.png?raw=true)

In a classification task using a CNN, understanding the discriminating behaviour of some layers is desirable. 
There are a number of ways that allow to investigate the dynamics of layers. 

With Class Activation Maps (CAM) we can inspect which areas of an image contribute the most to the final classification ([link to original paper by Zhou et al., 2016](https://arxiv.org/pdf/1512.04150.pdf)). 

**Grad-CAM** is a generalization of the CAM method and was introduced by [Selvaraju et al., 2017](https://arxiv.org/pdf/1610.02391.pdf).

The idea of Grad-CAM is to produce a hotmap of the most sensitive portion of an image for its classification using gradients. Following Selvaraju's notation, given:
- *c* the predicted class of image
- *y<sup>c</sup>* the model score 
- *A* the activation (feature) map of one layer of the network, indicized by k. E.g. *A<sup>k=6</sup>* refers to the 6-th channel of A. 
- ∂y<sup>c</sup>/∂A<sup>k</sup> the derivative of the score w.r.t. the k-th layer of the feature map A.

We then calculate:

<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha^c_k&space;=&space;\frac{1}{Z}\sum_i&space;\sum_j&space;\frac{\partial&space;y^c}{\partial&space;A^k_{ij}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha^c_k&space;=&space;\frac{1}{Z}\sum_i&space;\sum_j&space;\frac{\partial&space;y^c}{\partial&space;A^k_{ij}}" title="\alpha^c_k = \frac{1}{Z}\sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}" /></a> 

Every coeffient α<sub>k</sub> represents the (global averaged pooled) gradient of the predicted score y<sup>c</sup> with respect to the feature map A<sup>k</sup>.
These coeffients carry the 'importance' of the feature map as per the final score.

We then calculate a linear combination of the activation maps weighted by these coeffientients, and ReLu the result. 

<a href="https://www.codecogs.com/eqnedit.php?latex=L^c_{Grad-CAM}&space;=&space;ReLU&space;(\sum_k\alpha^c_k&space;A^k)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L^c_{Grad-CAM}&space;=&space;ReLU&space;(\sum_k\alpha^c_k&space;A^k)" title="L^c_{Grad-CAM} = ReLU (\sum_k\alpha^c_k A^k)" /></a>

Note that this map will have the spatial dimension of the activation maps A<sup>k</sup>, and not of the input image. 

You may consider to investigate different layers. Typically the last conv layer of a CNN architecture is chosen as it carries a decoded representation of the image while preserving its spatial information. 
