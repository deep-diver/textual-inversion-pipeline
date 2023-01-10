import tensorflow as tf

PATH="training_pipeline/tfrecords"

BATCH_SIZE = 4
AUTO = tf.data.AUTOTUNE

def parse_tfr(proto):
    feature_description = {
        "image": tf.io.VarLenFeature(tf.float32),
    }
    rec = tf.io.parse_single_example(proto, feature_description)
    image = tf.reshape(tf.sparse.to_dense(rec["image"]), (512, 512, 3))
    return {"image": image}

def prepare_dataset(PATH=PATH, batch_size=BATCH_SIZE):
    dataset = tf.data.TFRecordDataset(
        [filename for filename in tf.io.gfile.glob(f"{PATH}/*.tfrecord")],
        num_parallel_reads=AUTO,
    ).map(parse_tfr, num_parallel_calls=AUTO)

    dataset = dataset.batch(batch_size)

    return dataset

dataset = prepare_dataset()

for batch in dataset.take(2):
    print(batch["image"].shape)