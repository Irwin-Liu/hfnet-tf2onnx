# hfnet-tf2onnx
Change HFNet trained model from Tensorflow to ONNX

**HFNet**
HFNet tensorflow source code : [https://github.com/ethz-asl/hfnet]
HFNet trained model : [https://projects.asl.ethz.ch/datasets/doku.php?id=cvpr2019hfnet]

**Environment**
Tensorflow 1.14
ONNX 1.6
tf2onnx 1.6
numpy 1.16.4

**Usage of TF2ONNX**
Command line
```
python -m tf2onnx.convert --opset 11 --fold_const --saved-model ./models/hfnet --output ./models/savedmodel.onnx
```

Python API
```
python export_frozen.py
python frozen2onnx.py
```
