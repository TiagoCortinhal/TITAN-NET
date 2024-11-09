# 🌌 TITAN-Net: Semantics-Aware Multi-Modal Domain Translation 🌌

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](http://img.shields.io/badge/CS.CV-arXiv%2106.13974-B31B1B.svg)](https://arxiv.org/abs/2106.13974)

![TITAN-Net](images/TITANNET.gif)

**TITAN-Net** introduces a fresh, effective approach for bridging the gap between sensor modalities with different data formats! 🌉 By harnessing the power of scene semantics, TITAN-Net can, for the very first time, synthesize a panoramic color image directly from a 3D LiDAR point cloud. 

### ✨ Highlights

1. **Semantic Segmentation**: First, we segment the LiDAR point cloud and project it onto a spherical surface. 📡
2. **Modular Generative Framework**: Our approach is modular and generative, translating LiDAR segments into colorful panoramic images!
3. **Adversarial Learning**: Using a conditional GAN, we translate LiDAR segment maps to their camera image counterparts, creating a seamless color scene. 🎨
4. **Performance**: Quantitative evaluations on the Semantic-KITTI dataset show TITAN-Net outperforms strong baselines by a significant margin.

## 🔗 Models and Pretrained Weights

Below are links to the models and pretrained weights used in this project:

- **[TITAN-Net Weights](https://drive.google.com/file/d/1ypwqJEgwG90ATvbnw9zAubb6A-xZ8Zeh/view?usp=sharing)**
- **[SD-Net Weights](https://drive.google.com/file/d/1TLys-tZpqPrLXx7s8SImKgZy9wmtdsZY/view?usp=drive_link)** — from NVIDIA's [semantic-segmentation repository](https://github.com/NVIDIA/semantic-segmentation)
- **[SalsaNext Weights](https://drive.google.com/file/d/1utfzooTDAlV5M6XGvCE0-L-vbdLe_2rD/view?usp=share_link)** — from [Tiago Cortinhal's SalsaNext repo](https://github.com/TiagoCortinhal/SalsaNext)
- **[TITAN-Next Weights](https://drive.google.com/file/d/1fEKd9jHOV39smMATNm8x_EWH8L9QMSox/view?usp=drive_link)** — from [Tiago Cortinhal's TITAN-Next repository](https://github.com/TiagoCortinhal/TITAN-Next)

## 📹 Example Videos

Check out these example videos showing TITAN-Net in action, generating breathtaking RGB panoramic images! 🎥

### Full Panoramic RGB Generation
[![Panoramic RGB Generation](https://img.youtube.com/vi/eV510t29TAc/0.jpg)](https://www.youtube.com/watch?v=eV510t29TAc "Panoramic RGB Generation")

### Data Augmentation with Semantic Segmentation
See how easily we can use semantic segmentation maps for data augmentation in datasets like KITTI and Cityscapes!

[![KITTI Data Augmentation](https://img.youtube.com/vi/zR6Ix6YUhwI/0.jpg)](https://www.youtube.com/watch?v=zR6Ix6YUhwI "KITTI Data Augmentation")
[![Cityscapes Data Augmentation](https://img.youtube.com/vi/MHshzIIcirU/0.jpg)](https://www.youtube.com/watch?v=MHshzIIcirU "Cityscapes Data Augmentation")

## 📚 Citation

If you use **TITAN-Net** in your research, please consider citing our paper:

```bibtex
@misc{cortinhal2021semanticsaware,
  title={Semantics-aware Multi-modal Domain Translation: From LiDAR Point Clouds to Panoramic Color Images}, 
  author={Tiago Cortinhal and Fatih Kurnaz and Eren Aksoy},
  year={2021},
  eprint={2106.13974},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
