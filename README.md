# Tensorflow to TensorRT 

This project demonstrates how to convert a tensorflow model to tensorRT and show some simple benchmark using an experiment done using a GTX1060 GPU.
The reason this repo was started as I found most existing are either outdated or too technical to be understood by novices. The experiment follows most of the common experiment using a ResNet 50 Imagenet model. 

Step by step examples and visualization are available in the .ipynb file

* At this moment the repo is still using slightly outdated tftrt native conversion functions (create_inference_graph) instead of very recent TrtGraphConverter (April - May 2019) as some changes are still continuously being added at the time of writing. INT8 conversion is also not included as trt.calib_graph_to_infer_graph has been found to be removed from tf.nightly.

## Requirement ##
* Python (3.6)
* TensorRT (5.0.2.6, follow instructions on https://developer.nvidia.com/tensorrt)
* Tensorflow with TensorRT integration (> 1.13, recommended to use tf-nightly-gpu 1.14)
* Keras
* opencv-python (can be replaced with Keras image or pillow)

The notebook was run on a laptop with i7, 8GB RAM, and GTX 1060 GPU running ubuntu 16.04. The improvements might be more significant with better graphic cards.

## Visualizing graphs on tensorboard ##
If you intend to visualize the graphs before and after transformation on tensorboard, please get the code from official repo and import the .pb files after you export them from the notebook.
[Import pb to Tensorboard](https://github.com/rockchip-linux/tensorflow/blob/master/tensorflow/python/tools/import_pb_to_tensorboard.py)

```
tensorboard --logdir=/tmp/tensorflow_logdir
```

## References ##
[Keras to Tensorflow](https://github.com/jeng1220/KerasToTensorRT/blob/master/README.md)
[Official TensorRT on Tensorflow page](https://github.com/tensorflow/models/tree/master/research/tensorrt)
[Speeding up tensorflow inference by Nvidia team](https://medium.com/tensorflow/speed-up-tensorflow-inference-on-gpus-with-tensorrt-13b49f3db3fa)



