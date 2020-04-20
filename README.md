# LCMCG.Pytorch

This repo is the official implementation of ["Learning Cross-Modal Context Graph for Visual Grounding"](https://arxiv.org/pdf/1911.09042.pdf) (AAAI2020)
## Installation
Check [INSTALL.md](INSTALL.md) for installation instructions.

## pre-requirements
1. Pre-computed bounding boxes are extracted by using [FasterRCNN](https://github.com/facebookresearch/maskrcnn-benchmark) \
We use the config "e2e_faster_rcnn_R_50_C4_1x.yaml" to train the object detector on MSCOCO dataset.
2. Language graph extraction by using [SceneGraphParser](https://github.com/vacancy/SceneGraphParser)
3. Some pre-processing data, like sentence annotations, box annotations.
4. You need to create the './flickr_datasets' folder and put all annotation in it.
The pretrained object detector weights, language parsing results and annotations can be found here at baidu-disk (link:https://pan.baidu.com/s/1bYbGUsHcZJQHele87MzcMg  password:5ie6)


## training

1. You can train our model by running the scripts 
```bash
sh scripts/train.sh
```

""