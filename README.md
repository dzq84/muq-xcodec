# muq-xcodec

## Introduction

This project implements a **24k** **semantic codec** with **25tps per layer**, and a total of **8 layers**, designed specifically for music.

### Features
- **Efficient Encoding**: 24k sample rate with **25tps per layer**, and a total of **8 layers**.
- **Semantic Encoding**: Utilizes the xcodec 1.0 method to incorporate semantic information
- **Self-Supervised Learning**: Implements MuQ as a self-supervised feature extractor
- **Lightning Model**: easy deployment and distributed training.

### Model Availability

Due to repository size limitations, the model is hosted on Hugging Face. You can access it here:

[muq-xcodec model on Hugging Face](https://huggingface.co/ZheqiDAI/muq-xcodec)

### Demo

You can listen to our demo, which includes results at both 100tps and 200tps. The demo showcases the model's performance at different time resolution settings.

### Enviroment

```
conda create --name muqxcodec --file environment.yml
```

### Train

```
python train.py --config config/config.yaml
```

### Inference

For detailed usage of the codec, please refer to the **inference** script

```
python inference.py
```

### citation

If you are interested in this work, please cite the following papers:

```
@misc{ye2024codecdoesmatterexploring,
      title={Codec Does Matter: Exploring the Semantic Shortcoming of Codec for Audio Language Model}, 
      author={Zhen Ye and Peiwen Sun and Jiahe Lei and Hongzhan Lin and Xu Tan and Zheqi Dai and Qiuqiang Kong and Jianyi Chen and Jiahao Pan and Qifeng Liu and Yike Guo and Wei Xue},
      year={2024},
      eprint={2408.17175},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2408.17175}, 
}

@misc{zhu2025muqselfsupervisedmusicrepresentation,
      title={MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization}, 
      author={Haina Zhu and Yizhi Zhou and Hangting Chen and Jianwei Yu and Ziyang Ma and Rongzhi Gu and Yi Luo and Wei Tan and Xie Chen},
      year={2025},
      eprint={2501.01108},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2501.01108}, 
}
```
