import tensorflow as tf
from tensorflow import keras
from utils import get_pos_ids, sample_from_encoder_outputs, get_timestep_embedding

class StableDiffusionFineTuner(keras.Model):
    def __init__(self, stable_diffusion, noise_scheduler, training_image_encoder, **kwargs):
        super().__init__(**kwargs)
        self.stable_diffusion = stable_diffusion
        self.noise_scheduler = noise_scheduler
        self.training_image_encoder = training_image_encoder

    def train_step(self, data):
        images, embeddings = data

        with tf.GradientTape() as tape:
            latents = sample_from_encoder_outputs(self.training_image_encoder(images))
            latents = latents * 0.18215

            noise = tf.random.normal(tf.shape(latents))
            batch_dim = tf.shape(latents)[0]

            timesteps = tf.random.uniform(
                 (batch_dim,), minval=0, maxval=self.noise_scheduler.train_timesteps, dtype=tf.int64
            )

            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # TODO(lukewood): figure out exactly what to feed to text encoder?
            encoder_hidden_state = self.stable_diffusion.text_encoder([embeddings, get_pos_ids()])

            # Not 100% certain if this is right
            # See https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_condition.py (def forward())
            # I did some more verification and I think this is, in fact, correct
            timestep_embeddings = tf.map_fn(fn=get_timestep_embedding, elems=timesteps, fn_output_signature=tf.float32)

            # Added [] around params
            noise_pred = self.stable_diffusion.diffusion_model(
                [noisy_latents, timestep_embeddings, encoder_hidden_state]
            )
            loss = self.compiled_loss(noise_pred, noise)
            loss = tf.reduce_mean(loss, axis=2)
            loss = tf.reduce_mean(loss, axis=1)
            loss = tf.reduce_mean(loss)

        trainable_weights = self.stable_diffusion.text_encoder.trainable_weights
        grads = tape.gradient(
            loss, trainable_weights
        )

        index_of_placeholder_token = tf.reshape(tf.where(grads[0].indices == 49408), ())
        condition = grads[0].indices == 49408
        condition = tf.expand_dims(condition, axis=-1)
        grads[0] = tf.IndexedSlices(
            values=tf.where(condition, grads[0].values, 0),
            indices=grads[0].indices,
            dense_shape=grads[0].dense_shape
        )

        self.optimizer.apply_gradients(
            zip(grads, trainable_weights)
        )
        return {"loss": loss}