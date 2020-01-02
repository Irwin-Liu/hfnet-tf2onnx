import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import tag_constants
tf.contrib.resampler
import tf2onnx

frozen_model = "./models/hf_frozen.pb"

with tf.Session() as sess:
    with tf.gfile.GFile(frozen_model, mode="rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph, opset = 11, input_names=[], output_names=["global_descriptor:0", "keypoints:0", "local_descriptors:0"])
        model_proto = onnx_graph.make_model("test")
        with open("hfnet.onnx", "wb") as f:
            f.write(model_proto.SerializeToString())
        print("done")
