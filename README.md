# LCMCG.Pytorch

This repo is the official implementation of ["Learning Cross-Modal Context Graph for Visual Grounding"](https://arxiv.org/pdf/1911.09042.pdf) (AAAI2020)
## Installation
Check [INSTALL.md](INSTALL.md) for installation instructions.

## pre-requirements
1. Download the flickr30k dataset in this [link](http://bryanplummer.com/Flickr30kEntities/)
2. Pre-computed bounding boxes are extracted by using [FasterRCNN](https://github.com/facebookresearch/maskrcnn-benchmark) \
We use the config "e2e_faster_rcnn_R_50_C4_1x.yaml" to train the object detector on MSCOCO dataset and extract the feature map at C4 layer.
3. Language graph extraction by using [SceneGraphParser](https://github.com/vacancy/SceneGraphParser)
4. Some pre-processing data, like sentence annotations, box annotations.
5. You need to create the './flickr_datasets' folder and put all annotation in it. I would highly recommend you to figure all 
the data path out in this project. You can refer this two file "maskrcnn_benchmark/config/paths_catalog.py" and "maskrcnn_benchmark/data/flickr.py" for details.

The pretrained object detector weights, language parsing results and annotations can be found here at baidu-disk (link:https://pan.baidu.com/s/1bYbGUsHcZJQHele87MzcMg  password:5ie6)


## training

1. You can train our model by running the scripts 
```bash
sh scripts/train.sh
```

""

## citation
If you are interested in our paper, please cite it.
```bash
@inproceedings{liu2019learning,
  title={Learning Cross-modal Context Graph for Visual Grounding},
  author={Liu, Yongfei and Wan, Bo and Zhu, Xiaodan and He, Xuming},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligenc}
  year={2020}
}
```