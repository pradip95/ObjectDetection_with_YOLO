"""
Created on Sunday, March 13
@author : Pradip Mehta
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import models
import loss
import data_loader


class Predict(object):
    def __init__(self, image_dir):
        self.image_dir = image_dir

    # method for preprocessing image data for prediction
    def preprocess_predict(self):
        image_path = [os.path.join(self.image_dir, x) for x in os.listdir(self.image_dir) if x[-3:] == "png"]
        images = []
        for image in image_path:
            image = plt.imread(image)
            print('input image shape: ', image.shape)
            #image = np.resize(image,(256, 384, 4))
            image = np.transpose(image[:, :, :3], (1, 0, 2))
            image.reshape((-1, image.shape[0], image.shape[1], image.shape[2]))
            images.append(image)
            print('input image reshaped: ', image.shape)
        return np.array(images)


# function for output prediction, input = image array from method preprocess_predict()
def predicting(data):
    # loading the trained model
    model = models.load_model("trained_models/yolo.model7", custom_objects={'loss_fn': loss.LossFunction.loss_fn})
    prediction = model.predict(data)

    return prediction


if __name__ == "__main__":
    # creating object of class Predict
    pred_obj = Predict('data/predict')
    image_pred = pred_obj.preprocess_predict()
    output_pred = predicting(image_pred)

    # fetching ground truth y_data
    data_load = data_loader.YOLODataset("data/predict", "data/predict", [384, 256], S=[[12, 8], [24, 16], [48, 32]],
                                        S_index=2, C=4)
    truth_labels = data_load.y_dataset_loader()

    # creating object of LossFunction class
    loss_obj = loss.LossFunction()
    total_loss, predicted_bbox, targeted_bbox, total_pred_lbl, gt_lbl = loss_obj.loss_fn(truth_labels, output_pred)

    print('predicted output shape: ', output_pred.shape)
    print('predicted label :', total_pred_lbl)
    print('ground truth label :', gt_lbl)

    # drawing predicted bbox
    data_load.draw_bbox(bboxes=predicted_bbox)
    data_load.draw_bbox(bboxes=targeted_bbox)
    data_loader.bb_intersection_over_union(predicted_bbox, targeted_bbox)


