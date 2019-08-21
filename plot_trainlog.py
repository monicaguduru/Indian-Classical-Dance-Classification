"""
Given a training log file, plot something.
"""
import csv
import matplotlib.pyplot as plt

def main(training_log):
    with open(training_log) as fin:
        reader = csv.reader(fin)
        # print(next(reader, None))
        # print(next(reader, None))
        next(reader, None)  # skip the header
        accuracies = []
        top_5_accuracies = []
        cnn_benchmark = []  # this is ridiculous
        for i in reader:#epoch 0 ,acc 1,categorical_accuracy 2,loss 3,mean_absolute_error 4,val_acc 5,val_categorical_accuracy 6,val_loss 7,val_mean_absolute_error 8
            
            accuracies.append(float(i[8]))
            top_5_accuracies.append(float(i[6]))
            cnn_benchmark.append(0.65)  # ridiculous
            # print(i)
            # print(acc)

        plt.plot(accuracies,label="val_mean_absolute_error")
        # plt.plot(top_5_accuracies)
        # plt.plot(cnn_benchmark,label="CNN Benchmark")
        plt.xlabel("No. of epochs")
        plt.ylabel("val_mean_absolute_error")
        plt.legend()
        # plt.plot(epoch)
        plt.show()

if __name__ == '__main__':
    training_log = 'data/logs/lstm-training-1551695343.8966677.log'   #log1(19 entries)
    # training_log = 'data/logs/lstm-training-1551688852.1953378.log'   #log2(6 entries)

    main(training_log)
