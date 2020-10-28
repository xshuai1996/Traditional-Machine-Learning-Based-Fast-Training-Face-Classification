# Traditional Machine Learning Based Fast-Training Face Classification
 
This work is an unofficial implementation of the paper [*An Interpretable Compression and Classification System: Theory and Applications*], which includes the following steps:
  1. Preprocessing (adjustable): Detect and crop the face, blacken the blackground, horizontal flip to augment the dataset, separate three color channels and equalize histogram individually.
  2. Feature extraction*: Seperate the image to local cuboids and run PCA for local cuboids in the same position of all images. This step will be executed until the shape of global cuboid goes to 1x1xC. This module is linear and inversable as a method of compression.
  3. Linear discriminant classifier: use the classical maximum a posteriori estimator
uses [*Saab transform*] for dimension reduction of local features. Based on an observation that the features extracted by 2. can be approximated by Gaussian distribution and applies pooled covariance matrix, the classifier will be written as a beautiful linear form. See original paper for formula derivation.

 *: Check [*Interpretable Convolutional Neural Networks via Feedforward Design*] for more mathematical explanation. 
 
## Examples of preprocessing results
First row shows original images in the [*Labeled Faces in the Wild (LFW)*], and second row shows corresponding preprocessed images.
![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/George_W_Bush_0157.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Jennifer_Lopez_0009.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Saddam_Hussein_0013.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Tony_Blair_0045.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/John_Negroponte_0012.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Junichiro_Koizumi_0012.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Lindsay_Davenport_0012.jpg)
![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/George_W_Bush_0157_F_57.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Jennifer_Lopez_0009_5.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Saddam_Hussein_0013_8.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Tony_Blair_0045_F_46.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/John_Negroponte_0012_F_9.jpg)  ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Junichiro_Koizumi_0012_0.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Lindsay_Davenport_0012_7.jpg)

## Experiment results
Accuracy under different numbers of classes:
\# classes | train acc | test acc
--- | --- | ---
20 | 94.53% | 86.49%
48| 92.80% | 81.56%
81 | 91.77% | 74.52% 
149 | 91.62% | 67.35%


Time consuming under different numbers of classes:
\# classes | \# train samples | train time | \# test samples | test time
---- | --- | --- | --- | ---
20 | 2541 | 1.47s | 851 | 0.08s
48 | 3611 | 2.09s | 1215 | 0.11s
81 | 4421| 2.49s | 1499 | 0.20s
149 | 5523 | 3.04s | 1893 | 0.18s

Note that since in this work the face segementation method is different with the paper ([*Face Parsing*] here), the accuracy is a little bit lower than the result presented in the paper. A few failure segmentation cases are shown below.
![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Amelie_Mauresmo_0021.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Atal_Bihari_Vajpayee_0021.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Britney_Spears_0008.jpg)  ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Gerhard_Schroeder_0073.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Jennifer_Capriati_0001.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Luiz_Inacio_Lula_da_Silva_0031.jpg)   
![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Amelie_Mauresmo_0021_F_8.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Atal_Bihari_Vajpayee_0021_7.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Britney_Spears_0008_0.jpg)  ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Gerhard_Schroeder_0073_F_13.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Jennifer_Capriati_0001_15.jpg) ![image](https://raw.githubusercontent.com/xshuai1996/Traditional-Machine-Learning-Based-Fast-Training-Face-Classification/master/preprocessing_examples/Luiz_Inacio_Lula_da_Silva_0031_15.jpg)

We can expect better performance if applying a better segmentation approach. For more experiments on top-x accuracy, image reconstruction, etc., please see [*An Interpretable Compression and Classification System: Theory and Applications*].

# Future Work

Though the paper did not mention specifically, the segmentation step is extremely important. The reason behind is, the [Saab transform] simply maximizes the variance of new features but is incapable to somehow form a mask or assign weights to discard background, which means whether a local cuboid is background or not it will be inevitably involved to form higher level features, and furthur put negative impact on the quality of high level features. From another angle, since both objects and background are compressed with same ratio in feature extraction, it wastes information to store background, which means useful features of objects might be lost. However, since background can't be distinguised in low level, it is challenging to obtain a "mask" in no-backward pass architecture, and the solution is still under development.

[*An Interpretable Compression and Classification System: Theory and Applications*]: <https://arxiv.org/abs/1907.08952>
[*Interpretable Convolutional Neural Networks via Feedforward Design*]: <https://arxiv.org/abs/1810.02786>
[*Face Parsing*]: <https://github.com/zllrunning/face-parsing.PyTorch>
[*Labeled Faces in the Wild (LFW)*]: <http://vis-www.cs.umass.edu/lfw/>
[Saab transform]: <https://arxiv.org/abs/1810.02786>