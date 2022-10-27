from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def compute_metrics(pred):
    """
    Compute general metrics during training: Accuracy and macro F1/Precision/Recall.
    :return:
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }