import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg


def _get_custom_metadata():
    """Define your custom categories and colors here"""
    custom_classes = [
        "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
    ]

    custom_colors = [
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170,
                                           30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [70, 130, 180],
        [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
        [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
    ]

    # For panoptic segmentation, you can add:
    # thing_classes = [class for countable objects]
    # stuff_classes = [class for amorphous regions]

    return {
        "stuff_classes": custom_classes,  # or separate into thing/stuff
        "stuff_colors": custom_colors,
    }


def register_custom_dataset(root):
    root = os.path.join(root, "custom_data")
    meta = _get_custom_metadata()

    DatasetCatalog.register(
        "custom_sem_seg",
        lambda: load_sem_seg(
            os.path.join(root, "annotations"),
            os.path.join(root, "images"),
            gt_ext='png',
            image_ext='jpg'
        )
    )

    MetadataCatalog.get("custom_sem_seg").set(
        image_root=os.path.join(root, "images"),
        seg_seg_root=os.path.join(root, "annotations"),
        evaluator_type="sem_seg",
        **meta
    )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_custom_dataset(_root)