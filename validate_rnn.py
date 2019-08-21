"""
Validate our RNN. Basically just runs a validation generator on
about the same number of videos as we have in our test set.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from models import ResearchModels
from data import DataSet
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix


n_classes = 6
# used in model evaluation
def generator_with_true_classes(generator,model):
    while True:
        x, y = next(generator)
        yield x, model.predict(x), y


def metric_calculation(generator, model, steps):
    Y_pred = []
    Y_true = []
    i = 0
    for x, y_pred, y_true in generator_with_true_classes(generator,model):
        Y_pred.extend(y_pred)
        Y_true.extend(y_true)
        i += 1
        if (i == steps + 1):
            break

    Y_pred = np.array(Y_pred)
    Y_true = (np.array(Y_true)).astype(int)
    Shape = Y_pred.shape
    Y_pred_ = np.argmax(Y_pred, axis=-1)
    Y_true_ = np.argmax(Y_true, axis=-1)
    print('this is the predicted labelaal:')
    print(Y_pred)

    print('this is the predicted labelaal:')
    print(Y_pred_)

    print('this is the true labelu:')
    #print(np.sum(Y_true,axis=0))
    print(Y_true)

    print('this is the true labelu:')
    print(Y_true_)
    print('Confusion Matrix : \n', confusion_matrix(Y_true_, Y_pred_), '\n')

    # target_names = ['NROI 0', 'ROI 1']
    target_names = ['Bharatnatyam','Kathak','Kuchipudi','Manipuri','Mohiniattam','Odissi']
    print(classification_report(Y_true_, Y_pred_, target_names=target_names))

    # class wise accuracy
    accuracy_class = np.full((n_classes), 0.)
    for i in range(n_classes):
        locs = np.array(np.where(Y_true_ == i))
        tp = np.count_nonzero(Y_true_[locs] == Y_pred_[locs])
        accuracy_class[i] = (tp / locs.shape[1]) * 100

    print('class_wise accuracy: ', accuracy_class)
    print('\n \n')





def validate(data_type, model, seq_length=40, saved_model=None,
             class_limit=None, image_shape=None):
    batch_size = 463

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    # # Evaluate!
    # results = rm.model.evaluate_generator(
    #    generator=val_generator,
    #    steps=10)
    #
    # print(results)
    # print(rm.model.metrics_names)

    print('Classification Metric for testing phase \n')
    metric_calculation(val_generator, rm.model, 0)

    #Y_pred = results
    # aa,bb=next(val_generator)
    # y_pred = np.argmax(Y_pred, axis=1)
    # print('Confusion Matrix')
    # print(confusion_matrix(val_generator.classes, y_pred))
    # print('Classification Report')
    # target_names = ['Bhangra','Bharatnatyam','Bihu','Kathak','Kuchipudi','Manipuri']
    # print(classification_report(val_generator.classes, y_pred, target_names=target_names))

def main():
    model = 'lstm'
    # saved_model = 'data/checkpoints/lstm-features.001-1.640.hdf5'  #bad   [ 9.5890411  0.         0.        37.5        0.         0.       ]
    # saved_model = 'data/checkpoints/lstm-features.001-1.661.hdf5'  #bad   [93.24324324  0.          0.          0.         21.91780822  0.        ] 
    # saved_model = 'data/checkpoints/lstm-features.009-1.314.hdf5'  #bad   [ 0.     0.   0.   0.   0.  94.59459459] 
    # saved_model = 'data/checkpoints/lstm-features.002-1.583.hdf5'  #bad   [10.46511628  0.  0.  0.  24.65753425  0.] 
    # saved_model = 'data/checkpoints/lstm-features.005-1.410.hdf5'   #bad    [ 0.          0.          0.         64.0625      0.         44.18604651] 
    # saved_model = 'data/checkpoints/mlp-features.001-11.459.hdf5'  #bad   [  0. 100.   0.   0.   0.   0.]
    # saved_model = 'data/checkpoints/mlp-features.021-4.906.hdf5'
    # saved_model = 'data/checkpoints/mlp-features.021-4.906.hdf5'
    # saved_model = 'data/checkpoints/lstm-features.005-1.399.hdf5'
    # saved_model = 'data/checkpoints/mlp-features.021-4.906.hdf5'
    # saved_model = 'data/checkpoints/mlp-features.021-4.906.hdf5'
    # saved_model = 'data/checkpoints/mlp-features.021-4.906.hdf5'
    # saved_model = 'data/checkpoints/mlp-features.021-4.906.hdf5'
    # saved_model = 'data/checkpoints/mlp-features.021-4.906.hdf5'
    # saved_model = 'data/checkpoints/conv_3d-images.001-1.877.hdf5'

    # resnet
    # saved_model = 'data1/checkpoints/lstm-features.004-1.279.hdf5'
    # inceptionV3
    saved_model = 'data/checkpoints/lstm-features.009-1.314.hdf5'
    # CON OF BOTH
    # saved_model = 'data2/checkpoints/lstm-features.004-1.277.hdf5'
    # saved_model = 'data2/checkpoints/lstm-features.004-1.360.hdf5'
    # saved_model = 'data2/checkpoints/lstm-features.004-1.314.hdf5'

    if model == 'conv_3d' or model == 'lrcn':
        data_type = 'images'
        image_shape = (80, 80, 3)
    else:
        data_type = 'features'
        image_shape = None

    validate(data_type, model, saved_model=saved_model,
             image_shape=image_shape, class_limit=None)

if __name__ == '__main__':
    main()
