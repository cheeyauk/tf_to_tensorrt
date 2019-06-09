# Author: Kee Chee Yau
# Last Modified : 9th June 2019
#
# This code is a util code for performing tensorRT optimization. Some code ideas were taken and simplified from the tensorRT repo and a few references given below:
#
# https://github.com/tensorflow/models/tree/master/research/tensorrt
# https://github.com/jeng1220/KerasToTensorRT
# https://developer.download.nvidia.com/devblogs/tftrt_sample.tar.xz

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt # for nightly build
#from tensorflow.contrib import tensorrt as trt # for tensorflow 1.13.1 or below, non nightly build 
import os
import copy
import numpy as np
from keras import backend as K
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
    
def write_graph_to_file(graph_name, graph_def, output_dir):
    """Write Frozen Graph file to disk."""
    output_path = os.path.join(output_dir, graph_name)
    with tf.gfile.GFile(output_path, "wb") as f:
        f.write(graph_def.SerializeToString())

# Frozen graph util for Keras to tensorflow pb conversion
class FrozenGraph(object):
    def __init__(self, model, shape):
        shape = (None, shape[0], shape[1], shape[2])
        x_name = 'image_tensor_x'
        with K.get_session() as sess:
            x_tensor = tf.placeholder(tf.float32, shape, x_name)
            K.set_learning_phase(0)
            y_tensor = model(x_tensor)
            y_name = y_tensor.name[:-2]
            graph = sess.graph.as_graph_def()
            graph0 = tf.graph_util.convert_variables_to_constants(sess, graph, [y_name])
            graph1 = tf.graph_util.remove_training_nodes(graph0)
            
        self.x_name = [x_name]
        self.y_name = [y_name]
        self.frozen = graph1 

        
# Loads frozen graph into Tensorflow
class TfModel(object):
    def __init__(self, graph, x_name,y_name):
        self.g = tf.Graph()
        with self.g.as_default():
            x_op, y_op = tf.import_graph_def(
                    graph_def=graph, return_elements=x_name + y_name)
            self.x_tensor = x_op.outputs[0]
            self.y_tensor = y_op.outputs[0]

        config = tf.ConfigProto(gpu_options=
                                tf.GPUOptions(per_process_gpu_memory_fraction=0.5,
                                              allow_growth=True))
        self.sess = tf.Session(graph=self.g, config=config)
    
    def infer(self, x):
        y = self.sess.run(self.y_tensor,feed_dict={self.x_tensor: x})
        return y

# This is used for calibrating of INT 8 tensorrt model. Unfortunately it is not ready now
def batch_from_image(file_name, batch_size, output_height=224, output_width=224,
                     num_channels=3):
    """Produce a batch of data from the passed image file.
    Args:
    file_name: string, path to file containing a JPEG image
    batch_size: int, the size of the desired batch of data
    output_height: int, final height of data
    output_width: int, final width of data
    num_channels: int, depth of input data
    Returns:
        Float array representing copies of the image with shape
        [batch_size, output_height, output_width, num_channels]
    """
    #image_array = preprocess_image(file_name, output_height, output_width, num_channels)
    image_array = image.load_img(file_name, target_size=(224, 224))
    image_array = image.img_to_array(image_array)
    image_array = preprocess_input(image_array)
    tiled_array = np.tile(image_array, [batch_size, 1, 1, 1])
    return tiled_array

# This is used for calibrating of INT 8 tensorrt model. Unfortunately it is not ready now
def get_iterator(data):
    """Wrap numpy data in a dataset."""
    dataset = tf.data.Dataset.from_tensors(data).repeat()
    return dataset.make_one_shot_iterator()

# This is used for calibrating of INT 8 tensorrt model. Unfortunately it is not ready now
def get_trt_graph_from_calib(converter, data, input_node, output_node,
                             num_loops=100):
    """Convert a TensorRT graph used for calibration to an inference graph."""
    converter.convert()
    def input_fn():
        iterator = get_iterator(data)
        return {input_node: iterator.get_next()}
    trt_graph = converter.calibrate(
        fetch_names=[output_node],
        num_runs=num_loops,
        input_map_fn=input_fn)
    return trt_graph

# convert tensorflow frozen graph to TensorRT
def tf_to_trt_graph(graph, y_name, batch_size, precision): 
    
    # New code in May 2019, not stable yet
        #converter = trt.TrtGraphConverter(
        #    input_graph_def=graph.frozen, nodes_blacklist=graph.y_name,
        #    max_batch_size=batch_size, max_workspace_size_bytes=1 << 30,
        #    precision_mode=precision)
        #self.tftrt_graph = converter.convert()
        
    if precision == "INT8":
        calib_graph = trt.create_inference_graph(
            graph,
            outputs=y_name,
            max_batch_size=batch_size,
            max_workspace_size_bytes=1 << 25,
            precision_mode=precision,
            minimum_segment_size=2)
        tftrt_graph=trt.calib_graph_to_infer_graph(calibGraph)
    else:
        tftrt_graph = trt.create_inference_graph(
            graph,
            outputs=y_name,
            max_batch_size=batch_size,
            max_workspace_size_bytes=1 << 25,
            precision_mode=precision,
            minimum_segment_size=2)
    return tftrt_graph
    
# Loads trt optimized graph
class TrtModel(TfModel):
    def __init__(self, trt_graph, batch_size,x_name,y_name, num_classes):
   
        self.num_classes = num_classes
        super(TrtModel, self).__init__(trt_graph,x_name,y_name)
        self.batch_size = batch_size

    def infer(self, x):
        num_tests = x.shape[0]
        y = np.empty((num_tests, self.num_classes), np.float32)
        batch_size = self.batch_size

        for i in range(0, num_tests, batch_size):
            x_part = x[i : i + batch_size]
            y_part = self.sess.run(self.y_tensor,
                                   feed_dict={self.x_tensor: x_part})
            y[i : i + batch_size] = y_part
        return y

def calculate_error_rate(result, ans):
    num_tests = ans.shape[0]
    error = 0
    for i in range(0, num_tests):
        a = np.argmax(ans[i])
        r = np.argmax(result[i])
        if (a != r) : error += 1
    
    error_rate = error/num_tests
    return error_rate
