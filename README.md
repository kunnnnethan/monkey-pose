# monkey-pose

Final project for UMN CSCI 5561 Computer Vision course

### Introduction

We chose the OpenMonkeyChallenge for our final project. The details of the challenge can be found in [here](http://openmonkeychallenge.com/challenge.html).

In this project, we proposed combining HRNet and the structure of CPMs together to create a model that has more complexity than either HRNet or CPMs. Our experiments show that our proposed method is able to surpass the performance of HRNet48. 

### Dataset

The study collected 112,360 images related to 26 species of primates. In our experiments, 66917 images are used for training, and 22306 images are used for validation and testing. More details of the dataset can be found in [here](https://competitions.codalab.org/competitions/34342).

Since we are using top-down methods for monkey pose estimation, we crop the region in which each monkey is relying for each image first according to the location of ground truth bounding boxes and place it in the center of the new image to specify which monkey should the model focus on.

### Model

<img src="https://github.com/kunnnnethan/monkey-pose/blob/main/figures/model.png" alt="model" width=120% height=120%/>

We take advantage of HRNet by replacing the features extractor with the entire HRNet and replacing convolution blocks in every CPMs stage with small-HRNets to further increase the complexity of the entire model. We create small-HRNet by discarding the fourth modularized block in the original HRNet and reducing the number of convolution layers in every modularized block. We use total three stages for our CPM- HRNet-combined model.

### Results

<img src="https://github.com/kunnnnethan/monkey-pose/blob/main/figures/train_acc.png" alt="train_acc" height="300"/><img src="https://github.com/kunnnnethan/monkey-pose/blob/main/figures/val_acc.png" alt="val_acc" height="300"/>

| Method | Accuracy (PCK@0.05) |
| -------- | -------- |
| CPM | 0.6216 |
| HRNet-w48 | 0.779 |
| CPHRNet-w48 | 0.7935 |
| CPRNetv2-w48 (our proposed method) | 0.7985 |

**Model Outputs:**

<img src="https://github.com/kunnnnethan/monkey-pose/blob/main/figures/result.png" alt="result" width=50% height=50%/>


### Usage

1. **Dataset** </br>
Download images and annotations from [OpenMonkeyChallenge](https://competitions.codalab.org/competitions/34342).
Afterwards, move train and validation annotation files to train and val file respectively. Your dataset file should look like the following:
    ```
    data/
    ├── train/
        ├── ...jpg
        └── train_annotation.json
    └── val/
        ├── ...jpg
        └── val_annotation.json
        ...
    ```
    (Optional) Run display_data.py to check if data are loaded correctly.
    ```
    python display_data.py
    ```

2. **Train** </br>
Modified arguments in [configs/train.yaml](https://github.com/kunnnnethan/monkey-pose/blob/main/configs/train.yaml) file before training. Several augmentation methods are provided as well. Set the following arguments to True if augmentations are needed.
    ```yaml
    preprocess:
        rotate: False
        scale: False
        horizontal_flip: False
        hsv: False
    ```
    Afterwards, simply run train.py
    ```
    python train.py
    ```

3. **Test** </br>
Similarly, modified arguments in [configs/test.yaml](https://github.com/kunnnnethan/monkey-pose/blob/main/configs/test.yaml) file before testing. Set the following argument to True if you want to visualize predicted result.
    ```yaml
    display_results: False
    ```
    Afterwards, run test.py
    ```
    python test.py
    ```
    You can also download weights that I trained for our project:
    * Model trained with augumentation: [CPHRNetv2](https://drive.google.com/uc?export=download&id=1V7JMERHAbamkJQ2n-eRwAXNOVtV8t3sm)
    



### References

[stefanopini/simple-HRNet](https://github.com/stefanopini/simple-HRNet)

**OpenMonkeyChallenge**
```
OpenMonkeyChallenge: Dataset and Benchmark Challenges for Pose Tracking of Non-human Primates
Yuan Yao, Abhiraj Mohan, Eliza Bliss-Moreau, Kristine Coleman, Sienna M. Freeman, Christopher J. Machado, Jessica Raper, Jan Zimmermann, Benjamin Y. Hayden, Hyun Soo Park
bioRxiv 2021.09.08.459549; doi: https://doi.org/10.1101/2021.09.08.459549
```

**Deep High-Resolution Representation Learning**
```
@inproceedings{sun2019deep,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={CVPR},
  year={2019}
}
```
