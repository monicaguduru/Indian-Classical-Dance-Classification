"""
Classify a few images through our CNN.
"""
import numpy as np
import operator
import random
import glob
import os.path
from data import DataSet
from processor import process_image
from keras.models import load_model

def main(nb_images=10):#number of test images 420 images
    """Spot-check `nb_images` images."""
    data = DataSet()
    model = load_model('data/checkpoints/inception.005-0.79.hdf5')
    # model = load_model('')
    # model = None

    # Get all our test images.
    images = glob.glob(os.path.join('data', 'test','**', '**','*.jpg'))
    print(nb_images)
    for _ in range(nb_images):
        print('-'*80)
        # Get a random row.
        sample = random.randint(0, len(images) - 1)
        image = images[sample]

        # Turn the image into an array.
        print(image)
        image_arr = process_image(image, (299, 299, 3))
        image_arr = np.expand_dims(image_arr, axis=0)  #Expand the shape of an array by inserting a new axis,
        # corresponding to a given position in the array shape. 
        #Useful when predicting a model based on a single input.

        # Predict.
        predictions = model.predict(image_arr)
        # print("hhhhhhhhhhhhhhhhhhhhhhhhhh")
        # print(predictions)
        # Show how much we think it's each one.
        label_predictions = {}
        for i, label in enumerate(data.classes):
            label_predictions[label] = predictions[0][i]

        sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)

        for i, class_prediction in enumerate(sorted_lps):
            # Just get the top five.
            # if i > 4:
            #     break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
            i += 1

if __name__ == '__main__':
    main()
