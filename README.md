# SelfSupervised_Document_Classification
Leveraging Self-Supervised learning to classify document scans.

| SimCLR |
| :---: | 
| ![SimCLR](ReadmeImages/Screenshot%202023-03-03%20at%204.19.39%20PM.png) |
\* *Figures are from its paper, SimCLR*

## What this script does
SimCLR is the simplest image-based self-supervised pretraining framework. It's not necessessarily the best, but is certainly elegant, easy to implement, and effective. This script investigates the how well the SimCLR algorithm performs on classifying document scans. 

Accuracy is judged using the linear classification protocol. This means that the SSL pre-training is done 100% without labels. After this, the backbone is frozen and a supervised linear classifier is trained. The accuracy with the linear classifier is what is reported. 

## Results:
![SimCLRResults](ReadmeImages/Screenshot%202023-07-04%20at%2010.10.11%20AM.png)

SimCLR (with a resnet-18 backbone) performed well, with 82% overall accuracy. This compares well to a fully supervised ResNet-18 (89%):

![SupervisedResnet18](ReadmeImages/Screenshot%202023-07-04%20at%2010.10.00%20AM.png)

T-SNE (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique used for visualizing high-dimensional data in a lower-dimensional space. It is commonly used in machine learning and data analysis to explore complex datasets and discover patterns or relationships between data points. Conceptually, it is this embedded space that was the input to the linear classifier. 

The T-SNE visualization of SimCLR is shown below. The classes are fairly well clustered. This indicates that the embedded space is sufficently differentated for downstream tasks. 

<object src="SimCLR_0.0001LR_InteractiveTSNE.html" width="100%" height="500px"></object>
