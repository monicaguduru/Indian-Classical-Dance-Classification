from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model  #
from keras.layers import Input
import numpy as np

class Extractor():
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model
        # D:/data/srilekha/edu/sem6/btp/btp/inception_v3_flow_kinetics.caffemodel
        if weights is None:
            # Get model with pretrained weights.
            base_model = InceptionV3(
                weights='imagenet',
                include_top=True #whether to include the fully-connected layer at the top of the network.
            )

            # We'll extract features at the final pool layer.
            self.model = Model(
                inputs=base_model.input,
                outputs=base_model.get_layer('avg_pool').output
            )
            # print("55555555555555555555555555555555555555555555555555555555")
            print(base_model.input)

        else:
            # Load the model first.
            self.model = load_model(weights)
            print(self.model)
            # Then remove the top so we get features not predictions.
            # From: https://github.com/fchollet/keras/issues/2371
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []

    def extract(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) #Expand the shape of an array.

#Insert a new axis that will appear at the axis position in the expanded array shape.
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)
        # print("777777777777777777777777777777777777777777777")
        # print(features)
        #print(features.size())
        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features
