#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import some libraries
import os

# import gzip
# import PIL.ImageOps
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

# from PIL import Image

physical_devices = tf.config.list_physical_devices('GPU')
print(f'GPU device availability: {physical_devices}')

outer_dim = 3
num_classes = 10
img_size = [512, 960]

# In[ ]:


(train_dataset_raw, val_dataset_raw), dataset_info = tfds.load(
    "glenda_coco2:4.0.0", split=["train[:95%]", "train[95%:]"], with_info=True,
    shuffle_files=True, data_dir="tensorflow_datasets"
)

num_samples = dataset_info.splits['train'].num_examples
print(f"Total number samples in GLENDA dataset: {num_samples}")
print(dataset_info)


# In[ ]:


def normalize_with_moments(x, axes=[0, 1], epsilon=1e-8):
    mean, variance = tf.nn.moments(x, axes=axes)
    x_normed = (x - mean) / tf.sqrt(variance + epsilon)  # epsilon to avoid dividing by zero
    return x_normed


def preprocess_data(sample):
    seg_data = tf.cast(sample['segmentation'], dtype=tf.float32)
    img_data = tf.cast(sample['image'], dtype=tf.float32)
    img_data = tf.image.resize(img_data, img_size)
    seg_data = tf.image.resize(seg_data, img_size)

    neg_updates = tf.tile([0], [tf.reduce_sum(tf.cast(seg_data < 0.5, dtype=tf.int32))])
    pos_updates = tf.tile([1], [tf.reduce_sum(tf.cast(seg_data >= 0.5, dtype=tf.int32))])
    seg_data = tf.tensor_scatter_nd_update(seg_data, tf.where(seg_data >= 0.5), tf.cast(pos_updates, dtype=tf.float32))
    seg_data = tf.tensor_scatter_nd_update(seg_data, tf.where(seg_data < 0.5), tf.cast(neg_updates, dtype=tf.float32))

    return normalize_with_moments(img_data), seg_data


# In[ ]:


from tensorflow.keras import backend as K


# -----------------------------------------------------#
#                Soft Dice coefficient                #
# -----------------------------------------------------#
def dice_soft(y_true, y_pred, smooth=0.00001):
    """Dice loss originates from Sørensen–Dice coefficient, which is a statistic developed in the 1940s to gauge the similarity between two samples.
    Variant: Classwise score calculation
    Credits documentation: https://github.com/mlyg

    Parameters
    ----------
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
        :param smooth:
        :param y_pred:
        :param y_true:
    """
    # Identify axis
    axis = identify_axis(y_true.get_shape())

    # Calculate required variables
    intersection = y_true * y_pred
    intersection = K.sum(intersection, axis=axis)
    y_true = K.sum(y_true, axis=axis)
    y_pred = K.sum(y_pred, axis=axis)

    # Calculate Soft Dice Similarity Coefficient
    dice = ((2 * intersection) + smooth) / (y_true + y_pred + smooth)

    # Obtain mean of Dice & return result score
    dice = K.mean(dice)
    return dice


# -----------------------------------------------------#
#                    Tversky loss                     #
# -----------------------------------------------------#
#                     Reference:                      #
#                Sadegh et al. (2017)                 #
#     Tversky loss function for image segmentation    #
#      using 3D fully convolutional deep networks     #
# -----------------------------------------------------#
# alpha=beta=0.5 : dice coefficient                   #
# alpha=beta=1   : jaccard                            #
# alpha+beta=1   : produces set of F*-scores          #
# -----------------------------------------------------#
def tversky_class(y_true, y_pred, smooth=0.000001):
    # Define alpha and beta
    alpha = 0.5
    beta = 0.5
    # Calculate Tversky for each class
    axis = identify_axis(y_true.get_shape())
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    tp = K.sum(y_true * y_pred, axis=axis)
    fn = K.sum(y_true * (1 - y_pred), axis=axis)
    fp = K.sum((1 - y_true) * y_pred, axis=axis)
    tversky_class = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)

    return tversky_class


def tversky(y_true, y_pred, smooth=0.000001):
    # Sum up classes to one score
    tversky = K.mean(tversky_class(y_true, y_pred, smooth))
    # Return Tversky
    return tversky


def tversky_loss(y_true, y_pred, smooth=0.000001):
    # Sum up classes to one score
    tversky = K.sum(tversky_class(y_true, y_pred, smooth), axis=[-1])
    # Identify number of classes
    n = K.cast(K.shape(y_true)[-1], 'float32')
    # Return Tversky
    return n - tversky


# -----------------------------------------------------#
#             Tversky & Crossentropy loss             #
# -----------------------------------------------------#
def tversky_crossentropy(y_truth, y_pred):
    # Obtain Tversky Loss
    tversky = focal_tversky_loss(y_truth, y_pred)
    # Obtain Crossentropy
    crossentropy = K.binary_crossentropy(y_truth, y_pred)
    crossentropy = K.mean(crossentropy)
    # Return sum
    return tversky + crossentropy


def focal_tversky_loss(y_true, y_pred, delta=0.7, gamma=0.75, smooth=0.000001):
    # Clip values to prevent division by zero error
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    axis = identify_axis(y_true.get_shape())
    # Calculate true positives (tp), false negatives (fn) and false positives (fp)
    tp = K.sum(y_true * y_pred, axis=axis)
    fn = K.sum(y_true * (1 - y_pred), axis=axis)
    fp = K.sum((1 - y_true) * y_pred, axis=axis)
    tversky_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
    # Average class scores
    return K.mean(K.pow((1 - tversky_class), gamma))


# -----------------------------------------------------#
#                     Subroutines                     #
# -----------------------------------------------------#
# Identify shape of tensor and return correct axes
def identify_axis(shape):
    if len(shape) == 5:
        return [1, 2, 3]
    elif len(shape) == 4:
        return [1, 2]
    # Exception - Unknown
    else:
        print(shape)
        raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


# In[ ]:


import tensorflow_addons as tfa

from tensorflow.keras.layers import Dropout, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model


class Plain:
    # ---------------------------------------------#
    #                Initialization               #
    # ---------------------------------------------#
    def __init__(self, activation='sigmoid', conv_layer_activation='lrelu',
                 batch_normalization=True, batch_normalization_params=None,
                 dropout=0, pooling=(1, 2, 2)):
        # Parse parameter
        if batch_normalization_params is None:
            batch_normalization_params = {'epsilon': 1e-5}
        self.activation = activation
        # Parse activation layer
        if conv_layer_activation == "lrelu":
            self.conv_layer_activation = LeakyReLU(alpha=0.1)
        # Batch normalization settings
        self.ba_norm = batch_normalization
        self.ba_norm_params = batch_normalization_params
        # Dropout params
        self.dropout = dropout
        # Adjust pooling step
        self.pooling = pooling
        # Create list of filters
        self.feature_map = [30, 60, 120, 240, 320]

    def create_model(self, n_labels):
        # Input layer
        inputs = Input(img_size + [outer_dim])
        # Start the CNN Model chain with adding the inputs as first tensor
        cnn_chain = inputs
        # Cache contracting normalized conv layers
        # for later copy & concatenate links
        contracting_convs = []

        # Contracting layers
        for i in range(0, len(self.feature_map)):
            neurons = self.feature_map[i]
            cnn_chain = conv_layer_2D(cnn_chain, neurons, self.conv_layer_activation,
                                      self.ba_norm, self.ba_norm_params, self.dropout, strides=1)
            cnn_chain = conv_layer_2D(cnn_chain, neurons, self.conv_layer_activation,
                                      self.ba_norm, self.ba_norm_params, self.dropout, strides=1)
            contracting_convs.append(cnn_chain)
            cnn_chain = MaxPooling2D(pool_size=(2, 2))(cnn_chain)

        # Middle Layer
        neurons = self.feature_map[-1]
        cnn_chain = conv_layer_2D(cnn_chain, neurons, self.conv_layer_activation,
                                  self.ba_norm, self.ba_norm_params, self.dropout, strides=1)
        cnn_chain = conv_layer_2D(cnn_chain, neurons, self.conv_layer_activation,
                                  self.ba_norm, self.ba_norm_params, self.dropout, strides=1)

        # Expanding Layers
        for i in reversed(range(0, len(self.feature_map))):
            neurons = self.feature_map[i]
            cnn_chain = Conv2DTranspose(neurons, (2, 2), strides=(2, 2),
                                        padding='same')(cnn_chain)
            cnn_chain = concatenate([cnn_chain, contracting_convs[i]], axis=-1)
            cnn_chain = conv_layer_2D(cnn_chain, neurons, self.conv_layer_activation,
                                      self.ba_norm, self.ba_norm_params, self.dropout, strides=1)
            cnn_chain = conv_layer_2D(cnn_chain, neurons, self.conv_layer_activation,
                                      self.ba_norm, self.ba_norm_params, self.dropout, strides=1)

        # Output Layer
        conv_out = Conv2D(n_labels, (1, 1), activation=self.activation)(cnn_chain)
        # Create Model with associated input and output layers
        model = Model(inputs=[inputs], outputs=[conv_out])
        # Return model
        return model


# ----------------------------------------------------#
#                   Subroutines 2D                    #
# ----------------------------------------------------#
# Convolution layer
def conv_layer_2D(input, neurons, activation, ba_norm, ba_params, dropout, strides=1):
    conv = Conv2D(neurons, (3, 3), padding='same', strides=strides)(input)

    if dropout:
        conv = Dropout(dropout)(conv)
    if ba_norm:
        conv = tfa.layers.InstanceNormalization(**ba_params)(conv)

    return activation(conv)


# In[ ]:

model = Plain().create_model(num_classes)
tf.keras.utils.plot_model(model, to_file='glenda_model/model.png', show_shapes=True)
def myprint(s):
    with open('glenda_model/modelsummary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

optimizer = tf.optimizers.Adam(learning_rate=5e-4)
model.compile(loss=tversky_crossentropy, optimizer=optimizer, metrics=[dice_soft])

latest_checkpoint = tf.train.latest_checkpoint("glenda_model")
model.load_weights(latest_checkpoint)

# In[ ]:


import matplotlib.pyplot as plt

plt.rc('font', size=10)  # controls default text sizes


def visualize_detections(image, boxes, classes, title, figsize=(17, 17), linewidth=1, color=[1, 1, 1]):
    #Visualize Detections
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    ax.set_title(title)

    for box, _cls in zip(boxes, classes):
        _cls_parts = _cls.split('.')
        text = f"{_cls_parts[0][:2].lower()}.{_cls_parts[1][:3].lower()}"

        y1, x1, y2, x2 = box
        w, h = x2 - x1, y2 - y1

        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            (x1 + x2) // 2,
            y2 - 10,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
    return ax


def dice_soft_np(y_true, y_pred, smooth=0.00001):
    axis = (0, 1)

    # Calculate required variables
    intersection = np.multiply(y_true, y_pred)
    intersection = np.sum(intersection, axis=axis)
    y_true = np.sum(y_true, axis=axis)
    y_pred = np.sum(y_pred, axis=axis)

    # Calculate Soft Dice Similarity Coefficient
    dice = np.divide((2 * intersection) + smooth, y_true + y_pred + smooth)

    return dice


# In[ ]:


from skimage.measure import label, regionprops
from sklearn.metrics import classification_report

cat_names = ['Adhesions.Dense', 'Adhesions.Filmy', 'Deep.Endometriosis', 'Ovarian.Chocolate Fluid',
             'Ovarian.Endometrioma', 'Superficial.Black', 'Superficial.Red', 'Superficial.Subtle',
             'Superficial.White', 'Background']

all_dice_scores = []
all_y_true, all_y_pred = [], []

for k, sample in enumerate(val_dataset_raw):
    img_data_orig = sample['image']
    seg_data_orig = sample['segmentation']

    seg_data = tf.cast(seg_data_orig, dtype=tf.float32)
    img_data = tf.cast(img_data_orig, dtype=tf.float32)
    img_data = tf.image.resize(img_data, img_size)
    seg_data = tf.image.resize(seg_data, img_size)

    neg_updates = tf.tile([0], [tf.reduce_sum(tf.cast(seg_data < 0.5, dtype=tf.int32))])
    pos_updates = tf.tile([1], [tf.reduce_sum(tf.cast(seg_data >= 0.5, dtype=tf.int32))])
    seg_data = tf.tensor_scatter_nd_update(seg_data, tf.where(seg_data >= 0.5), tf.cast(pos_updates, dtype=tf.float32))
    seg_data = tf.tensor_scatter_nd_update(seg_data, tf.where(seg_data < 0.5), tf.cast(neg_updates, dtype=tf.float32))

    pred_data = model.predict(normalize_with_moments(tf.expand_dims(img_data, 0)))
    prediction = np.where(pred_data > 0.9, 1, 0)[0]

    y_true = tf.reduce_max(seg_data, axis=[0, 1]).numpy().astype(int)
    y_pred = np.zeros((num_classes,), dtype=int)

    all_dice_scores.append(dice_soft_np(seg_data.numpy(), prediction))
    all_y_true.append(y_true)
    all_y_pred.append(y_pred)

    prediction_classes, prediction_boxes = [], []
    gt_classes, gt_boxes = [], []

    for i in range(num_classes - 1):
        ground_truth = label(seg_data_orig[..., i])
        prediction_cls = label(prediction[..., i])
        props_prediction = [p for p in regionprops(prediction_cls) if p.area > 100]
        props_gt = [p for p in regionprops(ground_truth)]

        prediction_boxes.extend([p.bbox for p in props_prediction])
        gt_boxes.extend([p.bbox for p in props_gt])

        prediction_classes.extend([cat_names[i]] * len(props_prediction))
        gt_classes.extend([cat_names[i]] * len(props_gt))

        if prediction_boxes:
            y_pred[i] = 1

    if k < 100:
        break
        # visualize_detections(img_data_orig, gt_boxes, gt_classes, "Ground Thruth")
        # visualize_detections(img_data, prediction_boxes, prediction_classes, "Model Predictions")

# In[ ]:


print("Complete DSC")
print(list(zip(cat_names, np.array(all_dice_scores).mean(axis=0))))

print("Classification report")
print(classification_report(np.array(all_y_true), np.array(all_y_pred), target_names=cat_names))

# In[ ]:


from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


############################################################
#  COCO Evaluation
############################################################
def evaluate_coco(model, dataset, coco, eval_type="bbox"):
    results, coco_image_ids = [], []

    for sample in tqdm(dataset):
        img_id = sample["id"].numpy().decode('utf-8')
        img_data_orig = sample['image']
        seg_data_orig = sample['segmentation']
        img_data_orig_shape = tf.shape(img_data_orig).numpy()

        seg_data = tf.cast(seg_data_orig, dtype=tf.float32)
        img_data = tf.cast(img_data_orig, dtype=tf.float32)
        img_data = tf.image.resize(img_data, img_size)
        seg_data = tf.image.resize(seg_data, img_size)

        pred_data = model.predict(normalize_with_moments(tf.expand_dims(img_data, 0)))
        prediction = np.where(pred_data > 0.9, 1, 0)[0]
        prediction_classes, prediction_boxes = [], []

        for i in range(num_classes - 1):
            prediction_cls = label(prediction[..., i])
            props_prediction = [p for p in regionprops(prediction_cls) if p.area > 100]
            props_gt = [p for p in regionprops(ground_truth)]

            prediction_boxes.extend([p.bbox for p in props_prediction])
            prediction_classes.extend([i * len(props_prediction)])

        ratio = (img_size[0] / img_data_orig_shape[0], img_size[1] / img_data_orig_shape[1])
        coco_image_ids.append(img_id)

        for _cls, bbox in zip(prediction_classes, prediction_boxes):
            y1, x1, y2, x2 = bbox
            w, h = x2 - x1, y2 - y1

            results.append({
                "image_id": img_id,
                "category_id": _cls,
                "bbox": [int(x1 / ratio[1]), int(y1 / ratio[0]), int(w / ratio[1]), int(h / ratio[0])],
                "score": 1.0
            })

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.params.iouThrs = np.linspace(.1, 0.95, int(np.round((0.95 - .1) / .05)) + 1, endpoint=True)
    cocoEval.params.maxDets = [1, 2, 5]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return cocoEval


# In[ ]:


path_common = "tensorflow_datasets"

coco_path = os.path.join(path_common, 'glenda_coco2', 'glenda_full_v3.json')
coco_annotation = COCO(annotation_file=coco_path)

coco_eval = evaluate_coco(model, val_dataset_raw, coco_annotation)

# In[ ]:
