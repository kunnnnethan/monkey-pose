# monkey-pose

### Display Data

After downloading train and validation data, put it under the "data/" file. Use display_data.py to visualize it to make sure there are no errors.

For example: </br>
<img src="https://i.imgur.com/hQs97A9.png" alt="example" height="300"/>


### Train

In the "configs/train.yaml" file, modify the following two arguments for specifying which type of model you are going to train and the name of that model. For the model type, you should specify either "CPM" or "HRNet".
For example:
```yaml
model_type: "CPM"
model_name: "test.model"
```

### Test

In the "configs/test.yaml" file, modify the following two arguments for specifying which type of model you are going to load and the name of that model. For the model type, you should specify either "CPM" or "HRNet".
For example:
```yaml
model_type: "CPM"
model_name: "test.model"
```

For testing, you can also download weights that I pretrained for the CPM and HRNet model. Make sure to put it under the "weights" folder after downloaded it.

[CPM](https://drive.google.com/uc?export=download&id=1jHdgP8fga_vg8KSEcYPp2k9lQZLiQ5kQ) </br>
[HRNet](https://drive.google.com/uc?export=download&id=1h0au3bN4VKak_jiqc1O_fSk2YlCRs2U2)
