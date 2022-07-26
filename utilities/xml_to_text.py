import os
import xml.etree.ElementTree as Et
import glob


# Function to get the data from XML Annotation
# noinspection PyTypeChecker
def extract_text(xml_path):

    # initialising dictionary
    box_info = {'bboxes': []}
    for xml_file in glob.glob(xml_path):
        print(xml_file)
        tree = Et.parse(open(xml_file))
        root = tree.getroot()

        for element in root:
            # get the filename
            if element.tag == "filename":
                box_info['filename'] = element.text

            # get the image size
            elif element.tag == "size":
                image_size = []
                for sub_element in element:
                    image_size.append(int(sub_element.text))

                box_info['image_size'] = tuple(image_size)

            # get details of bounding box
            elif element.tag == "object":
                bbox = {}
                for sub_element in element:
                    if sub_element.tag == "name":
                        bbox["class"] = sub_element.text

                    elif sub_element.tag == "bndbox":
                        for inner_sub_element in sub_element:
                            bbox[inner_sub_element.tag] = int(inner_sub_element.text)

                box_info['bboxes'].append(bbox)
                print(box_info)
    return box_info


# dictionary to map class to unique number
class_to_id = {"wirebrush": 0,
               "squeal": 1,
               "klacken": 2,
               "dyno": 3
               }


# convert the box_info to the yolo v3 format
def convert_label_to_yolo_v3(box_info):
    box_info_yolo = []

    for box in box_info["bboxes"]:
        class_id = class_to_id[box["class"]]

        # transform the bbox co-ordinates as per yolo v3 format
        box_center_x = (box["xmin"] + box["xmax"]) / 2
        box_center_y = (box["ymin"] + box["ymax"]) / 2
        box_width = (box["xmax"] - box["xmin"])
        box_height = (box["ymax"] - box["ymin"])

        # normalize the bbox coordinates with image size
        image_width, image_height, image_depth = box_info["image_size"]
        box_center_x /= image_width
        box_center_y /= image_height
        box_width /= image_width
        box_height /= image_height

        # write the bbox details to the file
        box_info_yolo.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, box_center_x, box_center_y, box_width,
                                                                     box_height))

    # Name of the file which we have to save
    save_file_name = os.path.join("../data/sample", box_info["filename"].replace("png", "txt"))

    # Save the annotation to disk
    print("\n".join(box_info_yolo), file=open(save_file_name, "w"))
    print(box_info_yolo)
    return box_info_yolo


x = extract_text('data/sample/0737_fl_nr00290_chn0001_eew.xml')
convert_label_to_yolo_v3(x)
