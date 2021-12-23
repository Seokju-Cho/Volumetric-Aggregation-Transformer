[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cost-aggregation-is-all-you-need-for-few-shot/semantic-correspondence-on-spair-71k)](https://paperswithcode.com/sota/semantic-correspondence-on-spair-71k?p=cost-aggregation-is-all-you-need-for-few-shot)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cost-aggregation-is-all-you-need-for-few-shot/semantic-correspondence-on-pf-pascal)](https://paperswithcode.com/sota/semantic-correspondence-on-pf-pascal?p=cost-aggregation-is-all-you-need-for-few-shot)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cost-aggregation-is-all-you-need-for-few-shot/semantic-correspondence-on-pf-willow)](https://paperswithcode.com/sota/semantic-correspondence-on-pf-willow?p=cost-aggregation-is-all-you-need-for-few-shot)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cost-aggregation-is-all-you-need-for-few-shot/few-shot-semantic-segmentation-on-pascal-5i-1)](https://paperswithcode.com/sota/few-shot-semantic-segmentation-on-pascal-5i-1?p=cost-aggregation-is-all-you-need-for-few-shot)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cost-aggregation-is-all-you-need-for-few-shot/few-shot-semantic-segmentation-on-pascal-5i-5)](https://paperswithcode.com/sota/few-shot-semantic-segmentation-on-pascal-5i-5?p=cost-aggregation-is-all-you-need-for-few-shot)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cost-aggregation-is-all-you-need-for-few-shot/few-shot-semantic-segmentation-on-coco-20i-1)](https://paperswithcode.com/sota/few-shot-semantic-segmentation-on-coco-20i-1?p=cost-aggregation-is-all-you-need-for-few-shot)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cost-aggregation-is-all-you-need-for-few-shot/few-shot-semantic-segmentation-on-coco-20i-5)](https://paperswithcode.com/sota/few-shot-semantic-segmentation-on-coco-20i-5?p=cost-aggregation-is-all-you-need-for-few-shot)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cost-aggregation-is-all-you-need-for-few-shot/few-shot-semantic-segmentation-on-fss-1000-1)](https://paperswithcode.com/sota/few-shot-semantic-segmentation-on-fss-1000-1?p=cost-aggregation-is-all-you-need-for-few-shot)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cost-aggregation-is-all-you-need-for-few-shot/few-shot-semantic-segmentation-on-fss-1000-1)](https://paperswithcode.com/sota/few-shot-semantic-segmentation-on-fss-1000-1?p=cost-aggregation-is-all-you-need-for-few-shot)

## Cost Aggregation Is All You Need for Few-Shot Segmentation
For more information, check out project [[Project Page](https://seokju-cho.github.io/VAT/)] and the paper on [[arXiv](https://arxiv.org/abs/2112.11685)].


# Network

Our model VAT is illustrated below:

![alt text](./images/ARCH.png)

# Environment Settings
```
git clone https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer.git

cd Volumetric-Aggregation-Transformer

conda env create -f environment.yaml
```


## Preparing Few-Shot Segmentation Datasets
Download following datasets:

> #### 1. PASCAL-5<sup>i</sup>
> Download PASCAL VOC2012 devkit (train/val data):
> ```bash
> wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
> ```
> Download PASCAL VOC2012 SDS extended mask annotations from our [[Google Drive](https://drive.google.com/file/d/1uOI6VaI7rChOadKAfku8mVy_Xy_xF46j/view?usp=sharing)].

> #### 2. COCO-20<sup>i</sup>
> Download COCO2014 train/val images and annotations: 
> ```bash
> wget http://images.cocodataset.org/zips/train2014.zip
> wget http://images.cocodataset.org/zips/val2014.zip
> wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
> ```
> Download COCO2014 train/val annotations from our Google Drive: [[train2014.zip](https://drive.google.com/file/d/1vn2hhGM8XofaHxET-WadbZmxphInh2N-/view?usp=sharing)], [[val2014.zip](https://drive.google.com/file/d/1ey9Bevxd2OM4Uwf1sHb3lEhHHZ0tnuUR/view?usp=sharing)].
> (and locate both train2014/ and val2014/ under annotations/ directory).

> #### 3. FSS-1000
> Download FSS-1000 images and annotations from our [[Google Drive](https://drive.google.com/file/d/1bZ9zLeBBvZAfGSY_U760dMGgnRmFEbtq/view?usp=sharing)].

Create a directory '../Datasets_VAT' for the above three few-shot segmentation datasets and appropriately place each dataset to have following directory structure:

    ../                         # parent directory
    └── Datasets_VAT/
        ├── VOC2012/            # PASCAL VOC2012 devkit
        │   ├── Annotations/
        │   ├── ImageSets/
        │   ├── ...
        │   └── SegmentationClassAug/
        ├── COCO2014/           
        │   ├── annotations/
        │   │   ├── train2014/  # (dir.) training masks (from Google Drive) 
        │   │   ├── val2014/    # (dir.) validation masks (from Google Drive)
        │   │   └── ..some json files..
        │   ├── train2014/
        │   └── val2014/
        └── FSS-1000/           # (dir.) contains 1000 object classes
            ├── abacus/   
            ├── ...
            └── zucchini/

# Training

Training on PASCAL-5<sup>i</sup>:

      python train.py --config "config/pascal_resnet{50, 101}/pascal_resnet{50, 101}_fold{0, 1, 2, 3}/config.yaml"

Training on COCO-20<sup>i</sup>:

      python train.py --config "config/coco_resnet50/coco_resnet50_fold{0, 1, 2, 3}/config.yaml"

Training on FSS-1000:

      python train.py --config "config/fss_resnet{50, 101}/config.yaml"

# Evaluation
![alt text](./images/quantitative.png)

- Download pre-trained weights on [Link](https://drive.google.com/drive/folders/1bGaT7p_PvEAx-6HgSFBMx1slQLgCjsZ1?usp=sharing)

Result on PASCAL-5<sup>i</sup>:

      python test.py --load "/path_to_pretrained_model/pascal_resnet{50, 101}/pascal_resnet{50, 101}_fold{0, 1, 2, 3}/"

Result on COCO-20<sup>i</sup>:

      python test.py --load "/path_to_pretrained_model/coco_resnet50/coco_resnet50_fold{0, 1, 2, 3}/"

Results on FSS-1000:

      python test.py --load "/path_to_pretrained_model/fss_resnet{50, 101}/"

# Acknowledgement <a name="Acknowledgement"></a>

We borrow code from public projects (huge thanks to all the projects). We mainly borrow code from  [HSNet](https://github.com/juhongm999/hsnet). 
