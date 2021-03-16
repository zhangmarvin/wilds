import torch
import torchvision

from collections import OrderedDict

from torch.jit.annotations import Tuple, List, Dict, Optional
# bit of an importing abuse
from torchvision.models.detection.faster_rcnn import (AnchorGenerator,
                                                      FastRCNNPredictor,
                                                      GeneralizedRCNNTransform,
                                                      MultiScaleRoIAlign,
                                                      TwoMLPHead,
                                                      load_state_dict_from_url,
                                                      model_urls,
                                                      resnet_fpn_backbone)
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import (RegionProposalNetwork,
                                              RPNHead,
                                              concat_box_prediction_layers)


# All code copied and only slightly modified from torchvision detection models

class ModifiedFasterRCNN(GeneralizedRCNN):
    """
    torchvision FasterRCNN with a modified forward method to fit the WILDS interface.
    See https://github.com/pytorch/vision/blob/release/0.8.0/torchvision/models/detection/faster_rcnn.py
    for relevant documentation.
    """
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels specifying the number of output channels "
                "(assumed to be the same for all the levels)"
            )

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor is not specified")
        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        rpn = ModifiedRegionProposalNetwork(rpn_anchor_generator, rpn_head,
                                            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
                                            rpn_batch_size_per_image, rpn_positive_fraction,
                                            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)
        roi_heads = ModifiedRoIHeads(box_roi_pool, box_head, box_predictor,
                                     box_fg_iou_thresh, box_bg_iou_thresh,
                                     box_batch_size_per_image, box_positive_fraction,
                                     bbox_reg_weights,
                                     box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]

        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        super(ModifiedFasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)

    def forward(self, images):
        """
        See https://github.com/pytorch/vision/blob/master/torchvision/models/detection/generalized_rcnn.py#L43
        """
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])

        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        images, _ = self.transform(images)
        features = self.backbone(images.tensors)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        if self.training:
            proposals, objectness, pred_bbox_deltas, rpn_labels, rpn_regression_targets = \
                    self.rpn(images, features, targets)
            detections, class_logits, box_regression, roi_labels, roi_regression_targets = \
                    self.roi_heads(features, proposals, images.image_sizes, targets)
            # this is pretty hacky, but not sure if there is a better way to do this
            detections[0]['aux_outputs'] = {
                'objectness': objectness,
                'pred_bbox_deltas': pred_bbox_deltas,
                'rpn_labels': rpn_labels,
                'rpn_regression_targets': rpn_regression_targets,
                'class_logits': class_logits,
                'box_regression': box_regression,
                'roi_labels': roi_labels,
                'roi_regression_targets': roi_regression_targets,
            }
        else:
            proposals = self.rpn(images, features, targets)
            detections = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        return detections


class ModifiedRegionProposalNetwork(RegionProposalNetwork):
    """
    torchvision RPN with a modified forward method to fit the WILDS interface.
    See https://github.com/pytorch/vision/blob/master/torchvision/models/detection/rpn.py
    for relevant documentation.
    """
    def forward(self, images, features):
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)
        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
        return boxes, objectness, pred_bbox_deltas, anchors

#         if self.training:
#             assert targets is not None
#             labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
#             regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
#             return boxes, objectness, pred_bbox_deltas, labels, regression_targets
#         return boxes


class ModifiedRoIHeads(RoIHeads):
    """
    torchvision RoIHeads with a modified forward method to fit the WILDS interface.
    See https://github.com/pytorch/vision/blob/master/torchvision/models/detection/roi_heads.py
    for relevant documentation.
    """
    def forward(self, features, proposals, image_shapes):
        ### Significant amount of mask-related code removed ###
        ### Significant amount of keypoint-related code removed ###
        assert not (self.has_keypoint() or self.has_mask())

        if self.training:
            proposals, matched_idxs, labels, regression_targets = \
                    self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)
        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        boxes, scores, preds = self.postprocess_detections(class_logits, box_regression,
                                                           proposals, image_shapes)
        num_images = len(boxes)

        for i in range(num_images):
            result.append({"boxes": boxes[i], "labels": preds[i], "scores": scores[i]})

        if self.training:
            return result, class_logits, box_regression, labels, regression_targets
        return result


def faster_rcnn(num_classes=91, pretrained=True, progress=True, pretrained_backbone=False,
                trainable_backbone_layers=3, **kwargs):
    """
    Modification of torchvision.models.detection.fasterrcnn_resnet50_fpn
    See https://github.com/pytorch/vision/blob/release/0.8.0/torchvision/models/detection/faster_rcnn.py
    for relevant documentation.
    """
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        trainable_backbone_layers = 5

    if pretrained:
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = ModifiedFasterRCNN(backbone, num_classes=91, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'], progress=progress)
        model.load_state_dict(state_dict)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
