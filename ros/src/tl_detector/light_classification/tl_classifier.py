import rospy
import numpy as np
import tensorflow as tf
from styx_msgs.msg import TrafficLight
import os 


class TLClassifier(object):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        
        # Used pre built model (from https://github.com/ChristianSAW/CarND-Capstone/tree/master/ros/src/tl_detector/models/ssd_sim) 
        self.ssd_graph_file = dir_path + '/model/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.ssd_graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            # The input placeholder for the image.
            # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

            # The classification of the object (integer id).
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        
        self.sess = tf.Session(graph=self.detection_graph)
    
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        with self.detection_graph.as_default():
             scores, classes = self.sess.run([self.detection_scores, self.detection_classes],
                                              feed_dict = {self.image_tensor: np.expand_dims(image, axis = 0)})

        scores  = np.squeeze(scores)
        classes = np.squeeze(classes)
       
        confidence_cutoff = 0.7
        if scores is None:
            return TrafficLight.UNKNOWN
        if scores[0] >= confidence_cutoff:
            if classes[0] == 1:
                return TrafficLight.GREEN
            elif classes[0] == 2:
                return TrafficLight.RED
            elif classes[0] == 3:
                return TrafficLight.YELLOW


        
