import numpy as np


def compute_confusion_matrix(conf_mat, total_true_class_1):
    ncorr = conf_mat[0][0] + conf_mat[1][1]
    acc = ncorr / np.sum(conf_mat) * 100

    total_pred_class_1 = conf_mat[0][1] + conf_mat[1][1]
    if total_pred_class_1 == 0:
        precision = 100.0
    else:
        precision = conf_mat[1][1] * 1.0 / (total_pred_class_1) * 100
    if total_true_class_1 == 0:
        recall = 100.0
    else:
        recall = conf_mat[1][1] * 1.0 / (total_true_class_1) * 100
    fscore = 0.0
    if precision > 0.0 and recall > 0.0:
        fscore = 2 * precision * recall / (precision + recall)

    result_string = "Confusion matrix\n"
    result_string += "True\Pred\t0\t1\n"
    result_string += "\t0\t" + str(conf_mat[0][0]) + "\t" + str(conf_mat[0][1]) + "\n"
    result_string += "\t1\t" + str(conf_mat[1][0]) + "\t" + str(conf_mat[1][1]) + "\n"
    result_string += (
        "total items = " + str(np.sum(conf_mat)) + ", ncorrect = " + str(ncorr) + "\n"
    )
    result_string += "Accuracy = " + str(acc) + "%\n"
    result_string += "Precision = " + str(precision) + "%\n"
    result_string += "Recall = " + str(recall) + "%\n"
    result_string += "Fscore = " + str(fscore) + "%\n"
    return [acc, precision, recall, fscore, result_string]


def calculate_metrics(pred, targets, targets_w):
    result_preds = []
    result_targets = []
    conf_mat = np.zeros((2, 2), dtype=np.int32)

    for index in range(len(targets)):
        for index2 in range(len(targets[0])):
            if targets_w[index][index2] > 0.0:
                if pred[index][index2][0] < 0.5:
                    result_preds.append(0)
                else:
                    result_preds.append(1)
                result_targets.append(targets[index][index2])

    for i in range(len(result_preds)):
        conf_mat[result_targets[i]][result_preds[i]] += 1

    total_true_class_1 = conf_mat[1][0] + conf_mat[1][1]
    res = compute_confusion_matrix(conf_mat, total_true_class_1)

    return res

