## DL-benchmark

Little test/benchmark for TensorFlow and PyTorch. 

### Getting started
First, create a conda env :

```
conda create -n test python3=7
conda activate test
```

Then install each requirement.

```
pip install -r ./pytorch/requirements.txt
pip install -r ./tensorflow/requirements.txt
```

Then you can run both tests

```
python ./python/main.py
python ./tensorflow/main.py
```
If everything went well, you should not see any output in the app.log contained in booth folders. 

### How the test works
For both frameworks, we created some random images and labels and we tried to classify them. For Pytorch we are using a 'large' resnet34 model to stress the GPU a little more, while in TensorFlow we are just using a simple fc model. The training outputs are contained in `logs.csv` inside each folder.