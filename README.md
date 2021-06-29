# TITAN-NET

[![bla](https://img.shields.io/apm/l/vim-mode)](https://opensource.org/licenses/MIT)
[![bla](http://img.shields.io/badge/CS.CV-arXiv%2106.13974-B31B1B.svg)](https://arxiv.org/abs/2106.13974)


![TITANNET](images/TITANNET.gif)

In this work, we present a simple yet effective framework to address the domain translation problem between different sensor modalities with unique data formats. By relying only on the semantics of the scene, our modular generative framework can, for the first time, synthesize a panoramic color image from a given full 3D LiDAR point cloud. The framework starts with semantic segmentation of the point cloud, which is initially projected onto a spherical surface. The same semantic segmentation is applied to the corresponding camera image. Next, our new conditional generative model adversarially learns to translate the predicted LiDAR segment maps to the camera image counterparts. Finally, generated image segments are processed to render the panoramic scene images. We provide a thorough quantitative evaluation on the SemanticKitti dataset and show that our proposed framework outperforms other strong baseline models. 



# Example Videos
The following videos show the output of TITAN-NET and the generated RGB panoramic images.
[![Everything Is AWESOME](https://img.youtube.com/vi/eV510t29TAc/0.jpg)](https://www.youtube.com/watch?v=eV510t29TAc "Everything Is AWESOME")


In the following examples, we show how we can easily exploit the semantic-segmentation maps to perform data augmentation (KITTI and Cityscapes).

[![Everything Is AWESOME](https://img.youtube.com/vi/zR6Ix6YUhwI/0.jpg)](https://www.youtube.com/watch?v=zR6Ix6YUhwI "Everything Is AWESOME")
[![Everything Is AWESOME](https://img.youtube.com/vi/MHshzIIcirU/0.jpg)](https://www.youtube.com/watch?v=MHshzIIcirU "Everything Is AWESOME")



# Disclamer

This repository is for research purposes only, the use of this code is your responsibility.
