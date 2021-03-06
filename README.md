# BDG-Net: Boundary Distribution Guided Network for Accurate Polyp Segmentation

## 1. Overall Architecture

<p align="center">
    <img src="./overall.png"/> <br />
    <em> 
    Figure 1: Overall architecture of BDG-Net.
    </em>
    <img src="./table.png"/> <br />
</p>


## 2. Train/Test/Evaluate

### 2.1. Requirements 

torch

[pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)

[segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

[albumentations](https://github.com/albumentations-team/albumentations)

..

### 2.2 Download Necessary Data

Data can be found on [GoogleDrive](https://drive.google.com/drive/folders/1AQHCJ0kdOQl9j8OWfmXS4oeD6nQ9lUhd?usp=sharing), including train dataset, test dataset, and resultmap.

### 2.3. Train

Run `python train.py` to train on default setting.

To specify sigma, run `python train.py -s5`

### 2.4. Test

Run `python MyTest.py` to save the result map.

### 2.5. Evaluate 

The evaluation matlab code can be found in [PraNet](https://github.com/DengPingFan/PraNet), we use the same evaluation method as PraNet.

## 3. Citation

Please cite our paper if you find the work useful:
    
@inproceedings{10.1117/12.2606785,
	author = {Zihuan Qiu and Zhichuan Wang and Miaomiao Zhang and Ziyong Xu and Jie Fan and Linfeng Xu},
	booktitle = {Medical Imaging 2022: Image Processing},
	doi = {10.1117/12.2606785},
	editor = {Olivier Colliot and Ivana I{\v s}gum},
	keywords = {Polyp segmentation, Colorectal cancer, Colonoscopy, Deep learning},
	organization = {International Society for Optics and Photonics},
	pages = {792 -- 799},
	publisher = {SPIE},
	title = {{BDG-Net: boundary distribution guided network for accurate polyp segmentation}},
	url = {https://doi.org/10.1117/12.2606785},
	volume = {12032},
	year = {2022},
	Bdsk-Url-1 = {https://doi.org/10.1117/12.2606785}}
