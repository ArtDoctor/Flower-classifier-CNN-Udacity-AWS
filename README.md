# AI Programming with Python Project

**Instructions**:
Check out Image Cllassifier Project.ipynb if you want to get into code.

For training (example):
python train.py flowers --architecture 'resnet50'   

For predicting (example):
python predict.py model1.pth flowers/valid/58/image_02656.jpg --top_k_classes 3 --json_dict cat_to_name.json


Resources used:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
