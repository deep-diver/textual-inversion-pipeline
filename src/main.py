import tensorflow as tf
from tensorflow import keras
import keras_cv
from utils import traverse_layers
from prepare_text_encoder import prepare_text_encoder
from prepare_dataset import prepare_image_dataset, prepare_text_dataset

EPOCHS=50

stable_diffusion = keras_cv.models.StableDiffusion()
new_text_encoder = prepare_text_encoder(stable_diffusion)

image_dataset = prepare_image_dataset("../data")
text_dataset = prepare_text_dataset(stable_diffusion)

train_ds = tf.data.Dataset.zip((image_dataset, text_dataset))
train_ds = train_ds.batch(1).repeat(5).shuffle(20, reshuffle_each_iteration=True)

stable_diffusion.diffusion_model.trainable = False
stable_diffusion.decoder.trainable = False
stable_diffusion.text_encoder.trainable = True

stable_diffusion.text_encoder.layers[2].trainable = True

for layer in traverse_layers(stable_diffusion.text_encoder):
    if isinstance(layer, keras.layers.Embedding) or "clip_embedding" in layer.name:
        layer.trainable = True
    else:
        layer.trainable = False

new_text_encoder.layers[2].position_embedding.trainable=False

training_image_encoder = keras.Model(stable_diffusion.image_encoder.input, stable_diffusion.image_encoder.layers[-2].output)

noise_scheduler = NoiseScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", train_timesteps=1000
)
trainer = StableDiffusionFineTuner(stable_diffusion, noise_scheduler, training_image_encoder, name="trainer")

learning_rate = keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-4, decay_steps=train_ds.cardinality()*EPOCHS)
optimizer = keras.optimizers.Adam(weight_decay=0.01, learning_rate=learning_rate, epsilon=1e-8, global_clipnorm=10)

trainer.compile(
    optimizer=optimizer,
    loss=keras.losses.MeanSquaredError(reduction="none"),
)

trainer.fit(
    train_ds,
    epochs=EPOCHS,
)