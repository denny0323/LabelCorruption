# LabelCorruption

Pytorch code for the paper
@ Title : Noisy Label Correction for Classification of Wafer Bin Maps with Mixed-Type Defect Patterns
@ Author : SuminKim

### Abstract
Classification of defect patterns in wafer bin maps (WBMs) helps engineers detect process failures and identify their causes. 
In recent studies on WBMs, convolutional neural networks (CNNs) have demonstrated effective classification performance based 
on their high expressive power. However, previous studies have implicitly assumed that the labels of WBMs used for training CNNs are correct, 
even though labels are often incorrect. When trained on mislabeled data, CNNs with standard cross-entropy loss can easily overfit mislabeled samples,
leading to poor generalization for testing data. To overcome this issue, we propose a novel training algorithm called sample bootstrapping.

Sample bootstrapping identifies which samples have clean or noisy labels by using a two-component beta mixture model, and measures the uncertainty of each identified label.
Then, only the samples with low uncertainty of their estimated labels are selected to build mini-batches via weighted random sampling. 
Finally, CNNs are trained on the selected mini-batches with dynamic bootstrapping loss. In this manner, we can correct only the samples that are highly likely to have noisy labels and prevent the risk of false correction of correctly labeled samples. 
Experiments on WBM datasets demonstrate the effectiveness of the proposed method.


### Method
@ CrossEntropy : Standard training with the CrossEntropy loss in CNN
@ DFL : Discriminative feature learning and cluster-based defect label reconstruction for reducing uncertainty in wafer bin map labels [link](https://link.springer.com/article/10.1007/s10845-020-01571-4)
@ SELFIE : Pytorch version of the official code [SELFIE: Refurbishing Unclean Samples for Robust Deep Learning](http://proceedings.mlr.press/v97/song19b/song19b.pdf)
@ DivideMix : Pytorch version of the official code [DIVIDEMIX: LEARNING WITH NOISY LABELS AS SEMI-SUPERVISED LEARNING](https://openreview.net/pdf?id=HJgExaVtwr)
@ DYB : Dynamic bootstrap [Unsupervised Label Noise Modeling and Loss Correction](https://arxiv.org/pdf/1904.11238.pdf)
@ SB : Sample bootstrap (mine)
