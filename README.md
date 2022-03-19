# Speech Enhancement Project with Maxim Integrated
## Authors: Stella Cao, Renjie Liu, Huaxuan Wang, Curtis Zhuang
## Advisors: Batuhan Gundogdu, Utku Pamuksuz
### Best In Show Capstone Showcase

Capstone Project with Maxim Integrated working on building a light-weighted Deep Learning model that will enable the local speech enhancent processing on portable devices.

Speech processing is the study of digitally gathered speech signals and the processing methods of signals, and one specific area of speech processing is speech enhancement. Speech enhancement aims to increase speech quality through machine learning algorithms. Companies like Zoom use speech enhancement to improve its meetings’ qualities by reducing background noise. 

Maxim Integrated is a technology company that designs and manufactures technology equipment, and its new MAX78000 accelerator requires embedded speech enhancement algorithms. Because of hardware limitations, MAX78000 has very few parameter options and can only use convolutional neural networks. This paper specifically focuses on designing and optimizing speech enhancement algorithms compatible with MAX78000.

We proposed a small-U-Net CNN architecture specifically for the MAX78000 device from Maxim. We successfully deployed the proposed model on the device with a latency less than 10 msec and outstanding raising on SNR. Experiment results show that the proposed architecture has a significant improvement from the baseline model, and it performs the best among all our CNN architectures under the limitation of the MAX78000 device. We also show the performance of the proposed model on different contextual information and 6 types of noise and its compatibility with the MAX78000 accelerator.

Product Specification Page: https://github.com/MaximIntegratedAI/ai8x-training#limitations-of-max78000-networks
