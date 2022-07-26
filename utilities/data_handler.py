import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class MasterLoad(object):

    def __init__(self, img_path, label_path, image_size, batch_size):
        self.img_path = img_path
        self.label_path = label_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.image_data = []
        self.labels = []
        self.x_data = []
        self.y_data = []

    def process_image(self):

        for image_file in glob.glob(self.img_path):
            img_read = plt.imread(image_file)
            #print(img_read.shape)
            x = np.transpose(img_read, (1, 0, 2))
            #print(x.shape)
            self.image_data.append(x)
        x = np.asarray(self.image_data)

        x = x[:, :, :, :3]
        '''
        label_read = open(self.label_path, 'r')
        read = label_read.readlines()
        for line in read:
            label = [item.strip() for item in line.split(' ')[1:]]
            self.labels.append(label)
        y = np.asarray(self.labels) '''

        files = [os.path.join(self.label_path, x) for x in os.listdir(self.label_path) if x[-3:] == "txt"]
        for file in tqdm(files):
            label_read = open(file)
            read = label_read.readlines()
            for line in read:
                label = [item.strip() for item in line.split(' ')]
                if len(line) >= 0:
                    self.labels.append(label)
        y = np.asarray(self.labels)

        return x, y


if __name__ == "__main__":
    img_dir = 'data\\train\\*.png'
    label_dir = '../data/train_xml'
    a = MasterLoad(img_dir, label_dir, image_size=(384, 256), batch_size=1)

    x_data, y_data = a.process_image()
    print(y_data)
    print(x_data.shape)




































'''def pickle_image(self):

        """
        :return: None Creates a Pickle Object of DataSet
        """
        # Call the Function and Get the Data
        x, y = self.process_image()

        # Write the Entire Data into a Pickle File
        pickle_out = open('X_Data', 'wb')
        pickle.dump(x, pickle_out)
        pickle_out.close()

        # Write the Y Label Data
        pickle_out = open('Y_Data', 'wb')
        pickle.dump(y, pickle_out)
        pickle_out.close()

        print("Pickled Image Successfully ")
        return x, y

    def load_dataset(self):

        try:
            # Read the Data from Pickle Object
            X_Temp = open('X_Data', 'rb')
            x = pickle.load(X_Temp)

            Y_Temp = open('Y_Data', 'rb')
            y = pickle.load(Y_Temp)

            print('Reading Dataset from PIckle Object')

            return x, y

        except:
            print('Could not Found Pickle File ')
            print('Loading File and Dataset  ..........')

            x, y = self.pickle_image()
            return x, y'''