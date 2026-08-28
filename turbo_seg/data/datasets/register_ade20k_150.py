import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg


def _get_ade20k_150_meta():
    ade20k_150_classes = ["wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed ", "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion", "base", "box", "column", "signboard", "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television receiver", "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag"]

    ade20k_150_colors = [
        [197, 196, 23],  # wall
        [181, 87, 228],  # building
        [45, 37, 176],  # sky
        [21, 88, 64],  # floor
        [114, 237, 23],  # tree
        [222, 174, 10],  # ceiling
        [107, 11, 32],  # road
        [89, 173, 246],  # bed
        [155, 245, 212],  # windowpane
        [253, 77, 23],  # grass
        [11, 247, 152],  # cabinet
        [163, 140, 64],  # sidewalk
        [118, 24, 255],  # person
        [92, 6, 252],  # earth
        [12, 96, 230],  # door
        [241, 183, 84],  # table
        [249, 238, 178],  # mountain
        [236, 93, 193],  # plant
        [43, 21, 0],  # curtain
        [168, 120, 192],  # chair
        [104, 80, 118],  # car
        [173, 224, 31],  # water
        [100, 120, 254],  # painting
        [89, 56, 229],  # sofa
        [60, 251, 135],  # shelf
        [81, 72, 245],  # house
        [218, 210, 26],  # sea
        [63, 77, 192],  # mirror
        [20, 174, 65],  # rug
        [182, 218, 110],  # field
        [250, 251, 64],  # armchair
        [139, 224, 163],  # seat
        [27, 165, 104],  # fence
        [212, 57, 188],  # desk
        [112, 62, 2],  # rock
        [193, 149, 91],  # wardrobe
        [216, 50, 199],  # lamp
        [60, 60, 247],  # bathtub
        [4, 255, 242],  # railing
        [30, 11, 119],  # cushion
        [249, 32, 139],  # base
        [159, 86, 45],  # box
        [251, 139, 64],  # column
        [140, 237, 70],  # signboard
        [16, 101, 29],  # chest of drawers
        [83, 79, 187],  # counter
        [149, 207, 251],  # sand
        [217, 106, 66],  # sink
        [99, 161, 20],  # skyscraper
        [161, 10, 212],  # fireplace
        [61, 63, 192],  # refrigerator
        [122, 56, 70],  # grandstand
        [251, 131, 118],  # path
        [145, 185, 70],  # stairs
        [187, 212, 44],  # runway
        [199, 73, 113],  # case
        [150, 208, 179],  # pool table
        [63, 144, 70],  # pillow
        [6, 149, 64],  # screen door
        [176, 6, 205],  # stairway
        [28, 45, 29],  # river
        [108, 115, 187],  # bridge
        [179, 126, 130],  # bookcase
        [123, 17, 204],  # blind
        [232, 173, 114],  # coffee table
        [142, 241, 79],  # toilet
        [237, 254, 1],  # flower
        [103, 181, 187],  # book
        [202, 103, 204],  # hill
        [234, 218, 233],  # bench
        [85, 221, 24],  # countertop
        [62, 241, 149],  # stove
        [79, 122, 55],  # palm
        [192, 87, 21],  # kitchen island
        [137, 67, 164],  # computer
        [53, 154, 170],  # swivel chair
        [154, 3, 5],  # boat
        [86, 209, 125],  # bar
        [132, 114, 81],  # arcade machine
        [66, 181, 80],  # hovel
        [31, 97, 12],  # bus
        [212, 206, 168],  # towel
        [207, 32, 254],  # light
        [123, 115, 234],  # truck
        [59, 227, 89],  # tower
        [15, 191, 184],  # chandelier
        [39, 177, 249],  # awning
        [232, 153, 247],  # streetlight
        [217, 190, 226],  # booth
        [187, 150, 131],  # television receiver
        [154, 91, 230],  # airplane
        [247, 84, 123],  # dirt track
        [101, 39, 76],  # apparel
        [49, 165, 158],  # pole
        [167, 24, 48],  # land
        [190, 41, 31],  # bannister
        [125, 150, 168],  # escalator
        [169, 207, 246],  # ottoman
        [180, 167, 243],  # bottle
        [63, 233, 86],  # buffet
        [157, 199, 162],  # poster
        [16, 40, 167],  # stage
        [40, 151, 184],  # van
        [84, 113, 69],  # ship
        [217, 91, 165],  # fountain
        [89, 234, 127],  # conveyer belt
        [135, 162, 237],  # canopy
        [213, 84, 60],  # washer
        [218, 112, 74],  # plaything
        [210, 122, 130],  # swimming pool
        [163, 10, 17],  # stool
        [152, 178, 238],  # barrel
        [39, 145, 197],  # basket
        [136, 164, 163],  # waterfall
        [112, 103, 78],  # tent
        [218, 252, 187],  # bag
        [117, 98, 128],  # minibike
        [168, 128, 192],  # cradle
        [118, 59, 12],  # oven
        [30, 39, 151],  # ball
        [174, 0, 77],  # food
        [247, 77, 217],  # step
        [98, 117, 164],  # tank
        [195, 64, 196],  # trade name
        [181, 174, 228],  # microwave
        [125, 130, 81],  # pot
        [121, 49, 133],  # animal
        [250, 68, 41],  # bicycle
        [149, 32, 109],  # lake
        [97, 92, 40],  # dishwasher
        [24, 188, 61],  # screen
        [63, 217, 207],  # blanket
        [196, 19, 235],  # sculpture
        [141, 67, 99],  # hood
        [117, 71, 202],  # sconce
        [48, 172, 76],  # vase
        [68, 228, 71],  # traffic light
        [175, 114, 210],  # tray
        [219, 79, 106],  # ashcan
        [182, 190, 55],  # fan
        [88, 120, 120],  # pier
        [105, 149, 44],  # crt screen
        [187, 150, 131],  # plate
        [148, 175, 229],  # monitor
        [207, 197, 181],  # bulletin board
        [234, 77, 118],  # shower
        [244, 95, 38],  # radiator
        [196, 143, 14],  # glass
        [88, 2, 234],  # clock
        [44, 107, 49]  # flag
    ]

    ret = {
        "stuff_classes": ade20k_150_classes,
        "stuff_colors": ade20k_150_colors
    }
    return ret

def register_ade20k_150(root):
    root = os.path.join(root, "ADEChallengeData2016")
    meta = _get_ade20k_150_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("test", "images/validation", "annotations_detectron2/validation"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"ade20k_150_{name}_sem_seg"
        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='png', image_ext='jpg'))
        MetadataCatalog.get(name).set(image_root=image_dir, seg_seg_root=gt_dir, evaluator_type="sem_seg", ignore_label=255, **meta,)

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_ade20k_150(_root)
