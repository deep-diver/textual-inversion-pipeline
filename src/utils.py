import math
import tensorflow as tf

MAX_PROMPT_LENGTH = 77

def traverse_layers(layer):
    if hasattr(layer, "layers"):
        for layer in layer.layers:
            yield layer
    if hasattr(layer, "token_embedding"):
        yield layer.token_embedding
    if hasattr(layer, "position_embedding"):
        yield layer.position_embedding

def sample_from_encoder_outputs(outputs):
  mean, logvar = tf.split(outputs, 2, axis=-1)
  logvar = tf.clip_by_value(logvar, -30.0, 20.0)
  std = tf.exp(0.5 * logvar)
  sample = tf.random.normal(tf.shape(mean))
  return mean + std * sample


def get_timestep_embedding(timestep, dim=320, max_period=10000):
    half = dim // 2
    freqs = tf.math.exp(
        -math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
    )
    args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
    embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
    #embedding = tf.reshape(embedding, [1, -1])
    return embedding
    #return tf.repeat(embedding, batch_size, axis=0)

def get_pos_ids():
    return tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)    