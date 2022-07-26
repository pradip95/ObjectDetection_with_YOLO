"""
Created on Thursday Feb 15
@author : Pradip Mehta
"""

import os
import numpy as np


# import matplotlib.pyplot as plt
def len_annotation(folder):
    count = 0
    annotations = [os.path.join(folder, x) for x in os.listdir(folder) if x[-3:] == "txt"]
    for an in annotations:
        count += 1
    return count


def y_vector(folder):
    n = 48  # number of grid cells
    n1 = 32
    cell_width = 256
    cell_height = 384
    # y_data = []
    y_multiple = []
    count = 0
    annotations = [os.path.join(folder, x) for x in os.listdir(folder) if x[-3:] == "txt"]

    for an in annotations:
        count += 1
    print(count)
    # create empty tensor
    y = np.zeros((count, 48, 32, 9))

    #annotations = [os.path.join(folder, x) for x in os.listdir(folder) if x[-3:] == "txt"]
    annotation_count = 0

    for annotation in annotations:
        ann = open(annotation)

        #y = np.zeros((annotation_count, 32, 48, 9))
        line_count = len(ann.readlines())
        print('lines : ' + str(line_count))

        ann = open(annotation)
        read = ann.readlines()

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
            p2 = [cell_width / 256, 0.0]
            p4 = [0.0, cell_height / 384]
            p3 = [cell_width / 256, cell_height / 384]
            # print(p1,p2,p3,p4)

            # vectors v and w
            p01 = [p1[0] * (n - 1) / n + p2[0] * 1 / n, p1[1] * (n - 1) / n + p2[1] * 1 / n]
            p10 = [p1[0] * (n1 - 1) / n1 + p4[0] * 1 / n1, p1[1] * (n1 - 1) / n1 + p4[1] * 1 / n1]
            v = [p01[0] - p1[0], p01[1] - p1[1]]
            w = [p10[0] - p1[0], p10[1] - p1[1]]

            grid = []
            cell = []
            grid_array = []
            cell_array = []

            for j in range(0, n1 + 1):
                grid.append([])
                for i in range(0, n + 1):
                    p = [float(p1[0] + (i * v[0]) + (j * w[0])), float(p1[1] + (i * v[1]) + (j * w[1]))]
                    grid[j].append(p)

            # print('grid:')
            for i in range(0, n1 + 1):
                print(i)
                grid_array.append(grid)
                print(grid[i])

            for j in range(1, n1 + 1):  # i:x:col, j:y:line
                cell.append([])
                for i in range(1, n + 1):
                    cell[j - 1].append([grid[j - 1][i - 1], grid[j - 1][i], grid[j][i], grid[j][i - 1]])

            # print('cells:')
            for i in range(0, n1):
                for j in range(0, n):
                    cell_array.append(cell[i][j])
            # print(len(cell_array))

            # find the gridcell which contains bbox center
            for x in range(len(cell_array)):
                if cell_array[x][0][0] <= float(c_x) <= cell_array[x][2][0] and cell_array[x][0][1] <= float(c_y) \
                        <= cell_array[x][2][1]:
                    print('bbox_center_cell_number : ' + str(x))

                    center_cell_row_index = int(x / n)
                    print('center_cell_row_index : ' + str(center_cell_row_index))

                    center_cell_column_index = (x - int(x / n) * n) - 1
                    print('center_cell_column_index : ' + str(center_cell_column_index))

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
                    print('y_single : ' + str(y_single))

                    # insert bbox info y_single into y tensor
                    if line_count == 1:
                        y[annotation_count, center_cell_column_index, center_cell_row_index, :] = y_single
                        # print(y)
                        #y_data.append(y)
                        #print(np.array(y_data))
                        #print(y_data[0][0][6][0], y_data[0][1][6][2])
                    elif line_count > 1:
                        y_multiple.append(y_single)
                        # print(y_multiple)
                        y[annotation_count, center_cell_column_index, center_cell_row_index, :] = y_multiple[-1]
        #y_data.append(y)
        #print(np.array(y_data))
        #print(y)
        #print(y_data[0][0][4][6], y_data[1][1][4][6], y_data[1][1][3][6], y_data[1][1][2][6])
        annotation_count += 1
        #print(annotation_count)
    #print(y_data[0][0][6][2], y_data[1][1][6][2], y_data[1][0][4][6], y_data[1][1][3][6], y_data[1][1][2][6])
    #print(y[0][6][2], y[0][4][2], y[1][4][6], y[1][3][6], y[1][2][6], y[0][4][6], y[1][6][2])
    return print(np.array(y).shape)  # grid, cell_array, print(y), print(counter)


source_folder = 'data\\sample'
#len_annotation(source_folder)
y_vector(source_folder)

'''
def y_vec(folder):

    images = [os.path.join(folder, x) for x in os.listdir(folder) if x[-3:] == "png"]

    for image in images:
        img = plt.imread(image)
        cell_width = img.shape[0]
        cell_height = img.shape[1]
        y_vector(cell_width, cell_height)
    
'''

'''if num_lines == 0:
        y_data.append(y_)
    elif num_lines > 0:
        #print('line 1-')
        y_data.append(y_)
        #y_data.insert(y, -1)
    # print(num_lines)
    num_lines += 1'''
