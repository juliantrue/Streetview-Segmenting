import os
import sys
import json
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import imgaug

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Directory to save final model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class FacadesConfig(Config):
    """Configuration for training on dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "facades"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 12  # Background + NUM_CLASSES

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class FacadesDataset(utils.Dataset):

    def load_facades(self, dataset_dir, subset):
        """Load a subset of the Facades dataset.
        Expects COCO style dataset
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """

        # Train or validation dataset?
        assert subset in ["train", "val"]
        facades = COCO("{}/annotations/instances_facades_{}2019.json".format(dataset_dir, subset))
        image_dir = "{}/facades_{}2019".format(dataset_dir, subset)

        # Load all classes
        class_ids = sorted(facades.getCatIds())
        image_ids = list(facades.imgs.keys())
        for i in class_ids:
            self.add_class("facades", i, facades.loadCats(i)[0]["name"])
        for i in image_ids:
            self.add_image(
                "facades", image_id=i,
                path=os.path.join(image_dir, facades.imgs[i]['file_name']),
                width=facades.imgs[i]["width"],
                height=facades.imgs[i]["height"],
                annotations=facades.loadAnns(facades.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a facades image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "facades":
            return super(FacadesDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "facades.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:


                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(FacadesDataset, self).load_mask(image_id)

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train or Inference with Mask R-CNN .')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'inference'")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO style dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--inference_dir', required=False,
                        metavar="/path/to/inference_data/",
                        help='Path to data for inferencing')

    args = parser.parse_args()

    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == 'train':
        config = FacadesConfig()
    else:
        class InferenceConfig(FacadesConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
            class_names = ['background','background_2_electric_boogaloo',
                           'facade', 'molding', 'cornice', 'pillar', 'window',
                           'door', 'sill', 'blind', 'balcony', 'shop', 'deco']

        config = InferenceConfig()
    config.display()

    if args.command == 'train':
        model = modellib.MaskRCNN(mode='training', config=config,
                                  model_dir=MODEL_DIR)
        # Which weights to start with?
        init_with = "coco"  # imagenet, coco, or last

        if init_with == "imagenet":
            model.load_weights(model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            COCO_PATH = os.path.join(args.model, 'mask_rcnn_coco.h5')
            model.load_weights(COCO_PATH, by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                        "mrcnn_bbox", "mrcnn_mask"])
        elif init_with == "last":
            # Load the last model you trained and continue training
            model.load_weights(args.model, by_name=True)

    elif args.command == 'inference':
        model = modellib.MaskRCNN(mode='inference', config=config,
                                  model_dir=MODEL_DIR)

        model.load_weights(args.model, by_name=True)


    if args.command == 'train':
        # Load dataset
        dataset_train = FacadesDataset()
        dataset_train.load_facades(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val =  FacadesDataset()
        dataset_val.load_facades(args.dataset, "val")
        dataset_val.prepare()

        # Image augmentation
        augmentation = imgaug.augmenters.Sometimes(0.5, [
            imgaug.augmenters.Fliplr(0.5),
            imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
        ])

        print("\n**** Training network heads ****\n")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=2,
                    augmentation=augmentation,
                    layers='heads')

        print("\n**** Training all layers ****\n")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=3,
                    augmentation=augmentation,
                    layers="all")

        i = 0
        fn = "mask_rcnn_facades{}.h5".format(i)
        model_path = os.path.join(MODEL_DIR, fn)

        while(os.path.isfile(model_path)):
            i += 1
            fn = "mask_rcnn_facades{}.h5".format(i)
            model_path = os.path.join(MODEL_DIR, fn)

        model.keras_model.save_weights(model_path)

    elif args.command == 'inference':
        for f in os.listdir(args.inference_dir):
            img = cv2.imread(os.path.join(args.inference_dir,f))
            results = model.detect([img], verbose=1)
            r = results[0]
            visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                            config.class_names, r['scores'])
