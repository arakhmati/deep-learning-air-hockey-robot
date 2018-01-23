predictor_verifier.cc can be used to make sure that C++ version of Caffe2 loads init_net.pb and predict_net.pb correctly.
It does not check the neural network for the accuracy of its labels, but rather makes sure that all the weights are loaded correctly and the output is not full of 'nan' values.

To use it:
```
cp predictor_verifier.cc /pathToCaffe2/caffe2/binaries/predictor_verifier.cc
cd /pathToCaffe2/caffe2/binaries/
make
./binaries/predictor_verifier --init_net=/pathToInitNet/init_net.pb --predict_net=/pathToPredictNet/predict_net.pb
```