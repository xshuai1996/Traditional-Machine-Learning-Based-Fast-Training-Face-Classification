# Traditional Machine Learning Based Fast-Training Face Classification
 
This work is an unofficial implementation of the paper [*An Interpretable Compression and Classification System: Theory and Applications*], which includes the following steps:
  1. Preprocessing (adjustable): Detect and crop the face, blacken the blackground, horizontal flip to augment the dataset, separate three color channels and equalize histogram individually.
  2. Feature extraction*: Seperate the image to local cuboids and run PCA for local cuboids in the same position of all images. This step will be executed until the shape of global cuboid goes to 1x1xC. This module is linear and inversable as a method of compression.
  3. Linear discriminant classifier: use the classical maximum a posteriori estimator
uses [*Saab transform*] for dimension reduction of local features. Based on an observation that the features extracted by 2. can be approximated by Gaussian distribution and applies pooled covariance matrix, the classifier will be written as a beautiful linear form. See original paper for formula derivation.

 *: Check [*Interpretable Convolutional Neural Networks via Feedforward Design*] for more mathematical explanation. 
 
## Examples of preprocessing results
![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Amelie_Mauresmo_0021.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Atal_Bihari_Vajpayee_0021.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Britney_Spears_0008.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/George_W_Bush_0157.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Gerhard_Schroeder_0073.jpg)

![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Amelie_Mauresmo_0021_F_8.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Atal_Bihari_Vajpayee_0021_7.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Britney_Spears_0008_0.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/George_W_Bush_0157_F_57.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Gerhard_Schroeder_0073_F_13.jpg)


Since in this work the face segementation method is different with the paper ([*Face Parsing*] here), the accuracy is a little bit lower than the result presented in the paper.

## Experiment results







[*An Interpretable Compression and Classification System: Theory and Applications*]: <https://arxiv.org/abs/1907.08952>
[*Interpretable Convolutional Neural Networks via Feedforward Design*]: <https://arxiv.org/abs/1810.02786>
[*Face Parsing*]: <https://github.com/zllrunning/face-parsing.PyTorch>