import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def import_coords(folder):
    annotations = [os.path.join(folder, x) for x in os.listdir(folder) if x[-3:] == "txt"]
    coords = []
    multiple_coords = []
    for annotation in annotations:
        ann = open(annotation)
        line_count = len(ann.readlines())       # counts number of lines in single annotation file
        print('lines : ' + str(line_count))

        ann = open(annotation)
        read = ann.readlines()      # reads each line in annotation file

        for line in read:
            box = [item.strip() for item in line.split(' ')]
            cell_id = box[0]
            c_x = box[1]
            c_y = box[2]
            width = box[3]
            height = box[4]
            info = np.array([float(cell_id), float(c_x), float(c_y), float(width), float(height)])
            if line_count == 1:
                coords.append(info)
            elif line_count > 1:
                multiple_coords.append(info)
                # print(y_multiple)
                coords[:] = multiple_coords[-1]
            #coords.append(info)
    print(coords)
    return print(np.array(coords).shape)

folder = 'data\\sample'
import_coords(folder)


'''       
def draw_bboxes(bbox, **kwargs):
    # optional arguments
    colormap = kwargs.get('colormap', None)  # dict keys: class labels, values: valid colors
    bbox_format = kwargs.get('bbox_format', 'xywh')
    image_shape = kwargs.get('image_shape', None)
    xticks = kwargs.get('xticks', None)
    yticks = kwargs.get('yticks', None)

    fig, ax = plt.subplots()

    # check size 
    if type(bbox[0]) != list:  # only a single bbox provided
        bbox = [bbox]  # put bbox into a list

    xmax = 0.0
    ymax = 0.0
    for bb in bbox:

        # color for current class label
        try:
            color = colormap[bb[0]]
        except:
            color = 'red'

        if bbox_format == 'xyxy':
            # convert into [x, y, w, h] for plotting
            bb[2] = bb[2] - bb[0]
            bb[3] = bb[3] - bb[1]

        # place bounding box
        rect = patches.Rectangle((bb[1], bb[2]), bb[3], bb[4], linewidth=3, edgecolor=color, facecolor='none',
                                 alpha=0.8)
        ax.add_patch(rect)

        # plot center of bounding box as +
        plt.plot(bb[0] + bb[2] / 2, bb[1] + bb[3] / 2, color=color, marker='+')

        # add the label w/o probability
        if (len(bb) == 5) and type(bb[4] == str):
            plt.text(bb[0] + bb[2], bb[1] + bb[3], bb[4],
                     fontsize=10,
                     color='white',
                     horizontalalignment='right',
                     verticalalignment='bottom',
                     bbox=dict(facecolor=color, edgecolor=color, alpha=0.8))
        # add the label w/ probability
        elif (len(bb) == 6) and type(bb[4] == str) and type(bb[5] == float):
            plt.text(bb[0] + bb[2], bb[1] + bb[3], bb[4] + ' (' + str(bb[5]) + ')',
                     fontsize=10,
                     color='white',
                     horizontalalignment='right',
                     verticalalignment='bottom',
                     bbox=dict(facecolor=color, edgecolor=color, alpha=0.8))

        # update max axis extents
        if image_shape is None:
            xmax = np.max([xmax, bb[0] + bb[2]])
            ymax = np.max([ymax, bb[1] + bb[3]])

    if image_shape is None:
        plt.xlim([0, xmax * 1.5])
        plt.ylim([0, ymax * 1.5])
    else:
        plt.xlim([0, image_shape[0]])
        plt.ylim([0, image_shape[1]])

    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    return fig, ax'''