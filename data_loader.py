"""
Created on Tuesday, March 8
@author : Pradip Mehta
"""

import os
import pickle
import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class YOLODataset(object):
    def __init__(self, image_dir, label_dir, image_size, S, S_index, C):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.S = S
        self.S_index = S_index
        self.C = C

    def x_dataset_loader(self):
        image_path = [os.path.join(self.image_dir, x) for x in os.listdir(self.image_dir) if x[-3:] == "png"]
        image_array = []
        for image in image_path:
            image = plt.imread(image)
            # print(image)
            image = np.transpose(image[:, :, :3], (1, 0, 2))  # slicing the 4th channel depth and transposing to
            image_array.append(image)             # interchange height and width position
        # print(f'x_{self.image_dir}_dataset shape : ', np.array(image_array).shape)

        '''#x_train_pickle = open("data/pickled_data/x_train.pickle", "wb")
        #pickle.dump(np.array(image_array), x_train_pickle)
        #x_train_pickle.close()
        #x_test_pickle = open("data/pickled_data/x_test.pickle", "wb")
        #pickle.dump(np.array(image_array), x_test_pickle)
        #x_test_pickle.close()'''

        return np.array(image_array)

    def y_dataset_loader(self):
        count = 0
        y_multiple = []
        num_anchors = 1
        annotation_count = 0
        image_width = self.image_size[0]
        image_height = self.image_size[1]
        n_grid = [[12, 8], [24, 16], [48, 32]]

        label_path = [os.path.join(self.label_dir, x) for x in os.listdir(self.label_dir) if x[-3:] == "txt"]

        for annotation in label_path:
            count += 1  # for counting total number of annotation files in the folder
        #print('number of examples : ' + str(count))

        targets = np.zeros((count, self.S[self.S_index][0], self.S[self.S_index][1], num_anchors * (5 + self.C)))
        # print(targets.shape)
        n = n_grid[self.S_index]
        # print(n)
        # print(np.array(targets).shape)

        for annotation in label_path:
            ann = open(annotation)
            line_count = len(ann.readlines())  # counts number of lines in single annotation file
            # print('lines : ' + str(line_count))

            ann = open(annotation)
            read = ann.readlines()  # reads each line in annotation file

            for line in read:
                box = [item.strip() for item in line.split(' ')]
                cell_id = box[0]
                c_x = box[1]
                c_y = box[2]
                width = box[3]
                height = box[4]
                bbox = cell_id, c_x, c_y, width, height

                # bounding rectangle
                p1 = [0.0, 0.0]
                p2 = [image_width / 384, 0.0]
                p4 = [0.0, image_height / 256]
                p3 = [image_width / 384, image_height / 256]
                # print(p1,p2,p3,p4)

                # vectors v and w
                p01 = [p1[0] * (n[0] - 1) / n[0] + p2[0] * 1 / n[0], p1[1] * (n[0] - 1) / n[0] + p2[1] * 1 / n[0]]
                p10 = [p1[0] * (n[1] - 1) / n[1] + p4[0] * 1 / n[1], p1[1] * (n[1] - 1) / n[1] + p4[1] * 1 / n[1]]
                v = [p01[0] - p1[0], p01[1] - p1[1]]
                w = [p10[0] - p1[0], p10[1] - p1[1]]

                grid = []
                cell = []
                grid_array = []
                cell_array = []

                for j in range(0, n[1] + 1):
                    grid.append([])
                    for i in range(0, n[0] + 1):
                        p = [float(p1[0] + (i * v[0]) + (j * w[0])), float(p1[1] + (i * v[1]) + (j * w[1]))]
                        grid[j].append(p)

                # print('grid:')
                for i in range(0, n[1] + 1):
                    grid_array.append(grid)
                    # print(grid[i])

                for j in range(1, n[1] + 1):  # i:x:col, j:y:line
                    cell.append([])
                    for i in range(1, n[0] + 1):
                        cell[j - 1].append([grid[j - 1][i - 1], grid[j - 1][i], grid[j][i], grid[j][i - 1]])

                # print('cells:')
                for i in range(0, n[1]):
                    for j in range(0, n[0]):
                        cell_array.append(cell[i][j])
                # print(len(cell_array))

                # find the grid-cell which contains bbox center
                for x in range(len(cell_array)):
                    if cell_array[x][0][0] <= float(c_x) <= cell_array[x][2][0] and cell_array[x][0][1] <= float(c_y) \
                            <= cell_array[x][2][1]:

                        # print('bbox_center_cell_number : ' + str(x + 1))

                        center_cell_row_index = int(x / n[0])
                        # print('center_cell_row_index : ' + str(center_cell_row_index))

                        center_cell_column_index = (x - (int(x / n[0]) * n[0]))
                        # print('center_cell_column_index : ' + str(center_cell_column_index))

                        cell_w = cell_array[x][2][0] - cell_array[x][0][0]
                        # print(cell_w)

                        cell_h = cell_array[x][2][1] - cell_array[x][0][1]
                        # print(cell_h)

                        """
                        now create the sub-tensor that we will insert into the larger one. 
                        shape: 
                           [Pc, x, y, w, h, C1, C2, C3, C4]
                        """

                        # Pc is always one for targets
                        # we know x, y, w, h
                        y_single = np.array([1, float(c_x), float(c_y), float(box[3]), float(box[4])])

                        # still need to one-hot encode our label into [c1, c2, c3, c4]
                        c = np.zeros(4)
                        for index in range(len(c)):
                            if index == int(box[0]):
                                c[index] = 1

                        # insert c into y_single
                        y_single = np.append(y_single, c)
                        # print('y_single : ' + str(y_single))
                        # print(y_single)

                        # insert bbox info y_single into y tensor
                        if line_count == 1:
                            targets[annotation_count, center_cell_column_index, center_cell_row_index, :] = y_single

                        elif line_count > 1:
                            y_multiple.append(y_single)
                            # print(y_multiple)
                            targets[annotation_count, center_cell_column_index, center_cell_row_index, :] = y_multiple[-1]
            # target.append(targets)
            # print(y)
            annotation_count += 1
            # print('annotation_count : ' + str(annotation_count))

        '''#y_train_pickle = open("data/pickled_data/y_train.pickle", "wb")
        #pickle.dump(np.array(targets), y_train_pickle)
        #y_train_pickle.close()
        #y_test_pickle = open("data/pickled_data/y_test.pickle", "wb")
        #pickle.dump(np.array(targets), y_test_pickle)
        #y_test_pickle.close()'''
        return np.array(targets)  # grid, cell_array, print(y), print(counter)

    def bounding_box(self):
        label_path = [os.path.join(self.label_dir, x) for x in os.listdir(self.label_dir) if x[-3:] == "txt"]

        count = 0
        annotation_count = 0
        bboxes = []

        for files in label_path:
            count += 1  # for counting total number of annotation files in the folder
        # print('number of examples : ' + str(count))

        for annotation in label_path:
            ann = open(annotation)
            line_count = len(ann.readlines())  # counts number of lines in single annotation file
            # print('lines : ' + str(line_count))

            ann = open(annotation)
            read = ann.readlines()  # reads each line in annotation file

            for line in read:
                box = [item.strip() for item in line.split(' ')]
                bbox = np.array([float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])])
                bboxes.append(bbox)

        annotation_count += 1
        return np.array(bboxes)

    def draw_bbox(self, bboxes):
        global x, y, w, h
        multi_b = []
        image_path = np.array([os.path.join(self.image_dir, x) for x in os.listdir(self.image_dir) if x[-3:] == "png"])
        label_path = [os.path.join(self.label_dir, x) for x in os.listdir(self.label_dir) if x[-3:] == "txt"]
        start_bbox = 0
        count = 0

        for annotation in label_path:
            ann = open(annotation)
            line_count = len(ann.readlines())  # counts number of lines in single annotation file
            # print('lines : ' + str(line_count))

            width = 384
            height = 256

            step_w = width / self.S[self.S_index][0]
            step_h = height / self.S[self.S_index][1]

            stop_bbox = line_count + start_bbox
            #print('start_bbox :', start_bbox)
            #print('stop_bbox :', stop_bbox)
            for sub_label in range(start_bbox, stop_bbox):
                b = np.array(bboxes[sub_label])
                p_obj, x, y, w, h = b
                p_obj *= 1
                x *= 384
                y *= 256
                w *= 384
                h *= 256
                #print(b)
                if line_count > 1:
                    multi_b.append(b)
            # print(multi_b)
            # Create a Rectangle patch
            if line_count == 1:
                #print(b)
                rect = patches.Rectangle((x - w / 2, y - h / 2), w, h, linewidth=2, edgecolor='w',
                                         facecolor='none')
                #rect = patches.Rectangle(())
                plt.gca().add_patch(rect)
                plt.plot(x, y, color='w', marker='+')
                print('Bbox Coordinates:', x, y, w, h)
                for i in range(self.S[self.S_index][0]):
                    plt.vlines(i * step_w, 0, 256, colors='black', linewidth=0.5)
                    for j in range(self.S[self.S_index][1]):
                        plt.hlines(j * step_h, 0, 384, colors='black', linewidth=0.5)
                plt.xlim([0, 384])
                plt.ylim([256, 0])

            elif line_count > 1:
                #print(multi_b)

                for i in range(self.S[self.S_index][0]):
                    plt.vlines(i * step_w, 0, 256, colors='black', linewidth=0.5)
                    for j in range(self.S[self.S_index][1]):
                        plt.hlines(j * step_h, 0, 384, colors='black', linewidth=0.5)
                for r in range(len(multi_b)):
                    p_obj, x, y, w, h = multi_b[r]
                    #print(x)
                    p_obj *= 1
                    x *= 384
                    y *= 256
                    w *= 384
                    h *= 256
                    rect = patches.Rectangle((x - w / 2, y - h / 2),
                                             w, h, linewidth=2, edgecolor='r',
                                             facecolor='none')
                    print('Bbox Coordinates:', x, y, w, h)
                    plt.gca().add_patch(rect)
                    plt.plot(x, y, color='r', linewidth=2, marker='+')

                plt.xlim([0, 384])
                plt.ylim([256, 0])

            start_bbox += line_count
        im = mpimg.imread(image_path[count])
        fig = plt.imshow(im)
        img = plt.show()
        count += 1
        return fig


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    print(boxA, boxB)
    boxA[..., 1] = boxA[..., 1] - boxA[..., 3] / 2
    boxA[..., 2] = boxA[..., 2] - boxA[..., 4] / 2
    boxA[..., 3] = boxA[..., 1] + boxA[..., 3] / 2
    boxA[..., 4] = boxA[..., 2] + boxA[..., 4] / 2
    boxA[..., 1] *= 384
    boxA[..., 2] *= 256
    boxA[..., 3] *= 384
    boxA[..., 4] *= 256
    print('boxA : ', boxA[..., 1], boxA[..., 2], boxA[..., 3], boxA[..., 4])

    boxB[..., 1] = boxB[..., 1] - boxB[..., 3] / 2
    boxB[..., 2] = boxB[..., 2] - boxB[..., 4] / 2
    boxB[..., 3] = boxB[..., 1] + boxB[..., 3] / 2
    boxB[..., 4] = boxB[..., 2] + boxB[..., 4] / 2
    boxB[..., 1] *= 384
    boxB[..., 2] *= 256
    boxB[..., 3] *= 384
    boxB[..., 4] *= 256
    print('boxB : ', boxB[..., 1], boxB[..., 2], boxB[..., 3], boxB[..., 4])

    xA = max(boxA[..., 1], boxB[..., 1])
    yA = max(boxA[..., 2], boxB[..., 2])
    xB = min(boxA[..., 3], boxB[..., 3])
    yB = min(boxA[..., 4], boxB[..., 4])

    print(xA, yA, xB, yB)

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))


    print('intersection area : ', interArea)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[..., 3] - boxA[..., 1]) * (boxA[..., 4] - boxA[..., 2]))
    boxBArea = abs((boxB[..., 3] - boxB[..., 1]) * (boxB[..., 4] - boxB[..., 2]))
    print('boxAarea : ', boxAArea, 'boxBarea : ', boxBArea)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    print('iou: ', iou, 'iou %: ', iou*100)

    return iou


def test():
    dataset = YOLODataset(
        "data/predict",
        "data/predict",
        [384, 256],
        S=[[12, 8], [24, 16], [48, 32]],
        S_index=2,
        C=4
    )
    #dataset.x_dataset_loader()
    #dataset.y_dataset_loader()
    bboxes_array = dataset.bounding_box()
    dataset.draw_bbox(bboxes_array)

if __name__ == "__main__":
    test()
