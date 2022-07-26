"""
Created on Monday, March 14
@author : Pradip Mehta
"""

from tensorflow.keras.activations import sigmoid
from tensorflow.keras.losses import MSE, BinaryCrossentropy, CategoricalCrossentropy
import data_loader

class LossFunction(object):

    def __init__(self):
        self.bce = BinaryCrossentropy(from_logits=True)
        self.cce = CategoricalCrossentropy(from_logits=True)
        self.mse = MSE
        self.sigmoid = sigmoid
        self.obj_penalty = 1
        self.no_obj_penalty = 0.5
        self.bounding_box_loss_penalty = 10
        self.class_probability_loss_penalty = 1

    def loss_fn(self, targets, predictions):

        obj = targets[..., 0] == 1
        no_obj = targets[..., 0] == 0

        # compare targets[0] and predictions[0] for object == 0
        no_object_loss = self.bce(targets[..., 0:1][no_obj], predictions[..., 0:1][no_obj])
        print('no_object_loss :', no_object_loss)

        # compare targets[0] and predictions[0] for object == 1
        object_loss = self.bce(targets[..., 0:1][obj], predictions[..., 0:1][obj])
        print('object_loss :', object_loss)

        # bounding_box_loss = MSE  # compare targets[1:5] and predictions[1:5]
        bounding_box_loss = self.mse(targets[..., 1:5][obj], predictions[..., 1:5][obj])
        print('bounding_box_loss :', bounding_box_loss)

        # class_probability_loss = Categorical Cross entropy  # compare targets[5:] and predictions[5:]
        class_probability_loss = self.cce(targets[..., 5:][obj], predictions[..., 5:][obj])
        print('class_probability_loss :', class_probability_loss)

        losses = self.obj_penalty * object_loss + self.no_obj_penalty * no_object_loss + \
            self.bounding_box_loss_penalty * bounding_box_loss + \
            self.class_probability_loss_penalty * class_probability_loss

        print('losses :', losses)

        return losses, predictions[..., 0:5][obj], targets[..., 0:5][obj], predictions[..., 0:][obj], targets[..., 0:][obj]
        # uncomment if you want to draw bbox for predictions
