import torch
from wilds.common.utils import avg_over_groups, maximum
from wilds.common.metrics.metric import ElementwiseMetric, Metric, MultiTaskMetric

class Loss(Metric):
    def __init__(self, loss_fn, name=None):
        self.loss_fn = loss_fn
        if name is None:
            name = 'loss'
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        """
        Helper for computing element-wise metric, implemented for each metric
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
        Output:
            - element_wise_metrics (Tensor): tensor of size (batch_size, )
        """
        return self.loss_fn(y_pred, y_true)

    def worst(self, metrics):
        """
        Given a list/numpy array/Tensor of metrics, computes the worst-case metric
        Args:
            - metrics (Tensor, numpy array, or list): Metrics
        Output:
            - worst_metric (float): Worst-case metric
        """
        return maximum(metrics)
    
class ElementwiseLoss(ElementwiseMetric):
    def __init__(self, loss_fn, name=None):
        self.loss_fn = loss_fn
        if name is None:
            name = 'loss'
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred, y_true):
        """
        Helper for computing element-wise metric, implemented for each metric
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
        Output:
            - element_wise_metrics (Tensor): tensor of size (batch_size, )
        """
        return self.loss_fn(y_pred, y_true)

    def worst(self, metrics):
        """
        Given a list/numpy array/Tensor of metrics, computes the worst-case metric
        Args:
            - metrics (Tensor, numpy array, or list): Metrics
        Output:
            - worst_metric (float): Worst-case metric
        """
        return maximum(metrics)

class MultiTaskLoss(MultiTaskMetric):
    def __init__(self, loss_fn, name=None):
        self.loss_fn = loss_fn # should be elementwise
        if name is None:
            name = 'loss'
        super().__init__(name=name)

    def _compute_flattened(self, flattened_y_pred, flattened_y_true):
        if isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss):
            flattened_y_pred = flattened_y_pred.float()
            flattened_y_true = flattened_y_true.float()
        flattened_loss = self.loss_fn(flattened_y_pred, flattened_y_true)
        return flattened_loss

    def worst(self, metrics):
        """
        Given a list/numpy array/Tensor of metrics, computes the worst-case metric
        Args:
            - metrics (Tensor, numpy array, or list): Metrics
        Output:
            - worst_metric (float): Worst-case metric
        """
        return maximum(metrics)

class DetectionLoss(Metric):
    def __init__(self, name=None):
        if name is None:
            name = 'loss'
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        aux = y_pred[0]['aux_outputs']
        objectness, pred_bbox_deltas, rpn_labels, rpn_regression_targets = \
                aux['objectness'], aux['pred_bbox_deltas'], aux['rpn_labels'], aux['rpn_regression_targets']
        class_logits, box_regression, roi_labels, roi_regression_targets = \
                aux['class_logits'], aux['box_regression'], aux['roi_labels'], aux['roi_regression_targets']
        loss_objectness, loss_rpn_box_reg = self._compute_rpn_loss(
            objectness, pred_bbox_deltas, rpn_labels, rpn_regression_targets,
        )
        loss_classifier, loss_box_reg = self._compute_roi_loss(
            class_logits, box_regression, roi_labels, roi_regression_targets,
        )
        return loss_objectness + loss_rpn_box_reg + loss_classifier + loss_box_reg
