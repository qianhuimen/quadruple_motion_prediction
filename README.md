# quadruple_motion_prediction
Code for paper "A Quadruple Diffusion Convolutional Recurrent Network for Human Motion Prediction". Accepted by IEEE Transactions on Circuits and Systems for Video Technology.

# Dependencies

* h5py
* Tensorflow

# Download the data

Get the example human3.6m dataset on exponential map format.

```bash
git clone https://github.com/Yutasq/quadruple_motion_prediction.git
cd quadruple_motion_prediction
mkdir data
cd data
wget http://www.cs.stanford.edu/people/ashesh/h3.6m.zip
unzip h3.6m.zip
rm h3.6m.zip
cd ..
```

# Training and evaluations
Training from scratch: `python translate.py`  
Save poses: `python translate.py --sample --iterations 50000 --load 50000`  
Evaluation: To reproduce the results from our paper, run  `python evaluate.py`  
Visualization: `python forward_kinematics.py` The action type and seed can also be modified inside this file.

# Bibtex
```
@article{men20quadruple,
 author={Men, Qianhui and Ho, Edmond S. L. and Shum, Hubert P. H. and Leung, Howard},
 journal={IEEE Transactions on Circuits and Systems for Video Technology},
 title={A Quadruple Diffusion Convolutional Recurrent Network for Human Motion Prediction},
 year={2020},
 doi={10.1109/TCSVT.2020.3038145},
 publisher={IEEE},
}
```
