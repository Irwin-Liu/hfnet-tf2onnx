import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
tf.contrib.resampler

model_dir = "./models/hfnet"
output_frozen_path = "./models/hf_frozen.pb"

class HFNet:
    def __init__(self, model_dir, output_frozen_path):
        self.session = tf.Session()
        self.image_ph = tf.placeholder(tf.float32, shape=(None, None, 3))
        # self.image_ph = tf.placeholder(tf.float32, shape=(1, 224, 224, 1))

        net_input = tf.image.rgb_to_grayscale(self.image_ph[None])
        # net_input = self.image_ph
        tf.saved_model.loader.load(
            self.session, [tag_constants.SERVING], model_dir,
            clear_devices=True,
            input_map={'image:0': net_input})

        graph = tf.get_default_graph()

        # self.summary_writer = tf.summary.FileWriter("./", graph)

        frozen_graph = tf.graph_util.convert_variables_to_constants(
            self.session, graph.as_graph_def(),
            output_node_names=['global_descriptor', 'keypoints', 'local_descriptors']
        )
        frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph, ['global_descriptor', 'keypoints', 'local_descriptors'])

        with tf.gfile.GFile(output_frozen_path, mode='wb') as f:
            f.write(frozen_graph.SerializeToString())

hfnet = HFNet(model_dir, output_frozen_path)
