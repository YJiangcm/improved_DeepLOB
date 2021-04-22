# improved_DeepLOB

This repo contains my reimplementation and improvement of DeepLOB model. The original paper is "[DeepLOB - Deep Convolutional Neural Networks](https://arxiv.org/pdf/1808.03668.pdf)". 

I modified the model by using Dilated Convolution which could **raise the accuracy by around 2.5% accuracy with only less than 1,000 parameters added**. The model structure I use is depicted as follows:  


<img src="https://github.com/YJiangcm/improved_DeepLOB/blob/master/outputs/model_structure.png" width="500" height="450">

The training process is visualized as follows:

<img src="https://github.com/YJiangcm/improved_DeepLOB/blob/master/outputs/dilated_FI-2010%20Loss%20Graph.png" width="500" height="320">

<img src="https://github.com/YJiangcm/improved_DeepLOB/blob/master/outputs/dilated_FI-2010%20Accuracy%20Graph.png" width="500" height="320">
