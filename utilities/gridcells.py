import os
import matplotlib.pyplot as plt
import numpy as np

n = 13  # number of grid cells
grid = []
cell = []
grid_array = []
cell_array = []

counter = 0
num_lines = 0
folder = 'data/train'


def y_vector():
    # bounding rectangle
    p1 = [0.0, 0.0]
    p2 = [cell_width / 256, 0.0]
    p4 = [0.0, cell_height / 384]
    p3 = [cell_width / 256, cell_height / 384]
    # print(p1,p2,p3,p4)

    # vectors v and w
    p01 = [p1[0] * (n - 1) / n + p2[0] * 1 / n, p1[1] * (n - 1) / n + p2[1] * 1 / n]
    p10 = [p1[0] * (n - 1) / n + p4[0] * 1 / n, p1[1] * (n - 1) / n + p4[1] * 1 / n]
    v = [p01[0] - p1[0], p01[1] - p1[1]]
    w = [p10[0] - p1[0], p10[1] - p1[1]]

    for j in range(0, n + 1):
        grid.append([])
        for i in range(0, n + 1):
            p = [float(p1[0] + (i * v[0]) + (j * w[0])), float(p1[1] + (i * v[1]) + (j * w[1]))]
            grid[j].append(p)

    for i in range(0, n + 1):
        grid_array.append(grid)
        # print('grid:')
        # print(grid[i])

    for j in range(1, n + 1):  # i:x:col, j:y:line
        cell.append([])
        for i in range(1, n + 1):
            cell[j - 1].append([grid[j - 1][i - 1], grid[j - 1][i], grid[j][i], grid[j][i - 1]])

    # print('cells:')
    for i in range(0, n):
        for j in range(0, n):
            cell_array.append(cell[i][j])

    for i in range(len(cell_array)):
        if cell_array[i][0][0] <= float(c_x) <= cell_array[i][2][0] and cell_array[i][0][1] <= float(c_y) \
                <= cell_array[i][2][1]:
            # print(cell_array[i])
            # y_data.append([])
            y = [1, float(c_x), float(c_y), float(box[3]), float(box[4])]
            p_c = np.zeros(4)
            # print(int(box[0]))
            for index in range(len(p_c)):
                if index == int(box[0]):
                    p_c[index] = 1
                    # print(p_c)
                    y = np.append(y, p_c)
                    #y = np.array(y)

            return print(y)       # grid, cell_array, print(y), print(counter)


if __name__ == "__main__":
    images = [os.path.join(folder, x) for x in os.listdir(folder) if x[-3:] == "png"]
    for image in images:
        img = plt.imread(image)
        cell_width = img.shape[0]
        cell_height = img.shape[1]

    annotations = [os.path.join(folder, x) for x in os.listdir(folder) if x[-3:] == "txt"]
    annotations.sort()
    for annotation in annotations:
        annotation = open(annotation)
        read = annotation.readlines()
        # num_lines += 1
        # line_c = 1
        for line in read:
            box = [item.strip() for item in line.split(' ')]
            cell_id = box[0]
            c_x = box[1]
            c_y = box[2]
            width = box[3]
            height = box[4]
            c = cell_id, c_x, c_y, width, height
            y_vector()
            counter += 1
