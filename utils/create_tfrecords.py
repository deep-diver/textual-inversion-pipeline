import argparse
import os
import glob

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tqdm
from PIL import Image

def load_dataset(directory="training_pipeline/data", extension="jpeg"):
    files = glob.glob(f"{directory}/*.{extension}")
    images = [keras.utils.load_img(img) for img in files]

    return images


def resize_img(image: tf.Tensor, resize: int) -> tf.Tensor:
    resize = keras.layers.Resizing(height=512, width=512, crop_to_aspect_ratio=True)
    image = np.array(resize(image))
    return image


def process_image(image: Image, resize: int = 512) -> tf.Tensor:
    image = np.array(image)
    image = tf.convert_to_tensor(image)
    image = resize_img(image, resize)
    return image

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_tfrecord(image: Image):
    image = process_image(image)
    image = tf.reshape(image, [-1])  # flatten to 1D array

    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "images": _float_feature(image.numpy()),
            }
        )
    ).SerializeToString()

def write_tfrecords(root_dir, images):
    print(f"there are {len(images)} of images to packed into TFRecord")
    filename = os.path.join(
        root_dir, "textual_inversion.tfrecord"
    )

    with tf.io.TFRecordWriter(filename) as out_file:
        for i in tqdm.tnrange(len(images)):
            image = images[i]
            example = create_tfrecord(image)
            out_file.write(example)

def main(args):
    images = load_dataset()

    if not os.path.exists(args.root_tfrecord_dir):
        os.makedirs(args.root_tfrecord_dir, exist_ok=True)

    write_tfrecords(args.root_tfrecord_dir, images)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_tfrecord_dir",
        help="Root directory where the TFRecord will be serialized.",
        default="training_pipeline/tfrecords",
        type=str,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)