import numpy as np


def F1(y_pred, y_true):
    from keras import backend as K
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    f1 = 2*((precision*recall)/(precision+recall+K.epsilon()))
    return f1


def F1_calculation(predictions, labels):
    print("F1_calculation for sure labels only")
    idx = (labels == 1) | (labels == 0)  # sure only
    labels = labels[idx, ]
    predictions = predictions[idx, ]


    TP = np.sum(np.round(predictions * labels))
    PP = np.sum(np.round(labels))
    recall = TP / (PP + 1e-7)

    PP2 = np.sum(np.round(predictions))
    precision = TP/(PP2 + 1e-7)

    F1 = 2*((precision*recall)/(precision+recall+1e-7))

    print('Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(precision*100, recall*100, F1*100))

    return [precision, recall, F1]