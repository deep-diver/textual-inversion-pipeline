
from keras_cv import layers as cv_layers
import tensorflow as tf
from tensorflow import keras
import numpy as np

def prepare_text_encoder(stable_diffusion, tokenized_initializer_token="dog")
    # Have to get 1-index because `encode` adds start of text and end of text tokens
    tokenized_initializer = stable_diffusion.tokenizer.encode(tokenized_initializer_token)[1]
    #new_weights = stable_diffusion.text_encoder.embedding.token_embedding(
    #    tokenized_initializer
    #)
    # This is a hack that we should fix
    new_weights = stable_diffusion.text_encoder.layers[2].token_embedding(tf.constant(tokenized_initializer))

    ##
    new_vocab_size = len(stable_diffusion.tokenizer.vocab)

    # SAme hack here to get embedding layer from layers[2]
    old_token_weights = (
        stable_diffusion.text_encoder.layers[2].token_embedding.get_weights()
    )
    old_position_weights = (
        stable_diffusion.text_encoder.layers[2].position_embedding.get_weights()
    )

    old_token_weights = old_token_weights[0]
    new_weights = np.expand_dims(new_weights, axis=0)
    new_weights = np.concatenate([old_token_weights, new_weights], axis=0)

    # Have to set download_weights False so we can init (otherwise tries to load weights)
    new_encoder = keras_cv.models.stable_diffusion.TextEncoder(
        keras_cv.models.stable_diffusion.stable_diffusion.MAX_PROMPT_LENGTH, vocab_size=new_vocab_size, download_weights=False
    )
    # More layers[2] hacks
    for index, layer in enumerate(stable_diffusion.text_encoder.layers):
    if index == 2:
        continue
    new_encoder.layers[index].set_weights(layer.get_weights())


    new_encoder.layers[2].token_embedding.set_weights([new_weights])
    new_encoder.layers[2].position_embedding.set_weights(old_position_weights)

    stable_diffusion._text_encoder = new_encoder
    stable_diffusion._text_encoder.compile(jit_compile=True)
    return new_encoder
