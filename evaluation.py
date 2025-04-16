import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def evaluate_metrics(results, num_classes=2):
    """
    Calculate and return classification model metrics including Precision, Recall, F1 Score, AUC-ROC, and Confusion Matrix.
    
    Parameters:
        results (list of dict): A list of dictionaries containing model prediction results. 
        Each dictionary should have the following keys:
            - 'gt_label': Ground truth label (scalar)
            - 'pred_label': Predicted label (scalar)
            - 'pred_score': Prediction scores (numpy array)
        num_classes (int): Number of classes in the classification problem. Defaults to 2 (binary classification).

    Returns:
        dict: A dictionary containing the computed metrics: 'precision', 'recall', 'f1', 'auc_roc', 'conf_matrix'
    """
    
    # Extract true labels and predicted labels
    y_true = np.array([item['gt_label'].item() for item in results])  # Get scalar value
    y_pred = np.array([item['pred_label'].item() for item in results])  # Get scalar value
    y_scores = np.array([item['pred_score'].numpy() for item in results])  # Convert to numpy array

    # Calculate Precision, Recall, and F1 Score
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Calculate AUC-ROC
    if num_classes == 2:
        # Binary classification
        y_scores_flat = y_scores[:, 1]  # Take scores for the positive class
        auc_roc = roc_auc_score(y_true, y_scores_flat)
    else:
        # Multi-class classification
        # Calculate AUC-ROC for each class using one-vs-rest scheme
        auc_roc = roc_auc_score(y_true, y_scores, multi_class='ovr')

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Return a dictionary with computed metrics
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'conf_matrix': conf_matrix
    }


def extract_log_data(file_path):
    """
    Extract loss and accuracy information from the log data.
    
    Parameters:
    - logs: List of log data
    
    Returns:
    - loss: List of loss values from the training logs.
    - train_top1_acc: List of top-1 accuracy values from the training logs.
    - var_top1_acc: List of top-1 accuracy values from the validation logs.
    - var_mean1_acc: List of mean-1 accuracy values from the validation logs.
    - var_mean_ap: List of mean average precision values from the validation logs.
    - iters: List of iteration numbers from the training logs.
    - epochs: List of epoch numbers from the validation logs.
    - base_lr: List of base learning rates from the training logs.
    - lr: List of learning rates from the training logs.
    """

    # Read all lines from the log file and parse each line as a JSON object
    with open(file_path, 'r') as file:
        logs = [json.loads(line.strip()) for line in file]

    # Initialise lists to store extracted values
    loss = []
    train_top1_acc = []
    var_top1_acc = []
    var_mean1_acc = []
    var_mean_ap = []
    iters = []
    epochs = []
    base_lr = []
    lr = []

    # Process each log entry
    for log_entry in logs:
        # Check if the log entry is from the training mode
        if log_entry['mode'] == 'train':
            # Extract and append training metrics
            loss.append(log_entry['loss'])
            iters.append(log_entry['iter'])
            train_top1_acc.append(log_entry['top1_acc'])
            base_lr.append(log_entry['base_lr'])
            lr.append(log_entry['lr'])
        # Check if the log entry is from the validation mode
        if log_entry['mode'] == 'val':
            # Extract and append validation metrics
            var_top1_acc.append(log_entry['acc/top1'])
            var_mean1_acc.append(log_entry['acc/mean1'])
            var_mean_ap.append(log_entry['acc/mean_average_precision'])
            epochs.append(log_entry['epoch'])

    # Return the extracted metrics
    return loss, train_top1_acc, var_top1_acc, var_mean1_acc, var_mean_ap, iters, epochs, base_lr, lr


