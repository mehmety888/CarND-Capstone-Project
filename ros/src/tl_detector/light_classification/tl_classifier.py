import rospy
import numpy as np
import tensorflow as tf
from styx_msgs.msg import TrafficLight
import os 


class TLClassifier(object):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.ssd_graph_file = dir_path + '/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.ssd_graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.sess = tf.Session(graph=self.detection_graph)
        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        
    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)
    
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes
    
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        boxes, scores, classes = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                              feed_dict = {self.image_tensor: np.expand_dims(image, axis = 0)})

        boxes   = np.squeeze(boxes)
        scores  = np.squeeze(scores)
        classes = np.squeeze(classes)
        
        confidence_cutoff = 0.0
        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)
        
        if scores is None:
            print("none")
            return TrafficLight.UNKNOWN
        return classes[0]-1


        
