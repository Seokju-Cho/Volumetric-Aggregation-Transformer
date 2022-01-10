[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cost-aggregation-is-all-you-need-for-few-shot/semantic-correspondence-on-spair-71k)](https://paperswithcode.com/sota/semantic-correspondence-on-spair-71k?p=cost-aggregation-is-all-you-need-for-few-shot)</br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cost-aggregation-is-all-you-need-for-few-shot/semantic-correspondence-on-pf-willow)](https://paperswithcode.com/sota/semantic-correspondence-on-pf-willow?p=cost-aggregation-is-all-you-need-for-few-shot)</br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cost-aggregation-is-all-you-need-for-few-shot/semantic-correspondence-on-pf-pascal)](https://paperswithcode.com/sota/semantic-correspondence-on-pf-pascal?p=cost-aggregation-is-all-you-need-for-few-shot)

## Cost Aggregation Is All You Need for Few-Shot Segmentation (arXiv'21)
For more information, check out the paper on [[arXiv](https://arxiv.org/abs/2112.11685)].

# Environment Settings
```
git clone https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer.git
cd Volumetric-Aggregation-Transformer
git checkout semantic-matching

conda create -n VAT_semantic_matching python=3.8
conda activate VAT_semantic_matching

conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip install -U scikit-image
pip install tensorboardX termcolor timm tqdm requests pandas albumentations einops
```

# Training
Training on SPair-71k:

      python train.py --benchmark spair

Training on PF-PASCAL:

      python train.py --benchmark pfpascal --epochs 300 --step [150,200,250]

# Evaluation
- Download pre-trained weights on [Link](https://drive.google.com/drive/folders/1dKLHSmajNwlzSV5jwh_d9X8zZwvxvJxu?usp=sharing)
- All datasets are automatically downloaded into directory specified by argument `datapath`

Result on SPair-71k: (PCK 54.2%)

      python test.py --pretrained "/path_to_pretrained_model/spair/" --benchmark spair

Results on PF-PASCAL: (PCK 92.3%)

      python test.py --pretrained "/path_to_pretrained_model/pfpascal/" --benchmark pfpascal

# Acknowledgement <a name="Acknowledgement"></a>

We borrow code from public projects (huge thanks to all the projects). We mainly borrow code from  [DHPF](https://github.com/juhongm999/dhpf) and [GLU-Net](https://github.com/PruneTruong/GLU-Net). 

### BibTeX
If you find this research useful, please consider citing:
````BibTeX
@article{hong2021cost,
  title={Cost Aggregation Is All You Need for Few-Shot Segmentation},
  author={Hong, Sunghwan and Cho, Seokju and Nam, Jisu and Kim, Seungryong},
  journal={arXiv preprint arXiv:2112.11685},
  year={2021}
}
````
