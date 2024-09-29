import tensorflow as tf
import keras
from keras.initializers import GlorotNormal
from keras.layers import Dense, BatchNormalization, LeakyReLU, Dropout

learning_rate = 1e-5
batch_size = 50
latent_dim = 32
init_kernel = tf.keras.initializers.glorot_normal

# from tensorflow.keras.layers import SpectralNormalization
# from tensorflow.train import ExponentialMovingAverage

class Encoder(keras.Model):
    def __init__(self, z_dim: int, do_spectral_norm: bool):
        super().__init__()
        kernel_initializer = GlorotNormal(seed=500)

        if do_spectral_norm:
            self.dense1 = keras.layers.SpectralNormalization(
                Dense(64, kernel_initializer=kernel_initializer))
            self.leaky_relu1 = keras.layers.SpectralNormalization(
                LeakyReLU(0.2))
            self.dense2 = keras.layers.SpectralNormalization(
                Dense(z_dim, kernel_initializer=kernel_initializer))
        else:
            self.dense1 = Dense(64, kernel_initializer=kernel_initializer)
            self.leaky_relu1 = LeakyReLU(0.2)
            self.dense2 = Dense(z_dim, kernel_initializer=kernel_initializer)

    def call(self, x_input, training=False):
        z = self.dense1(x_input)
        z = self.leaky_relu1(z)
        return self.dense2(z)


class Generator(keras.Model):
    def __init__(self, x_dim: int):
        super().__init__()
        kernel_initializer = GlorotNormal(seed=500)

        self.dense1 = Dense(64, activation="relu", kernel_initializer=kernel_initializer)
        self.dense2 = Dense(128, activation="relu", kernel_initializer=kernel_initializer)
        self.dense3 = Dense(x_dim, kernel_initializer=kernel_initializer)

    def call(self, z_input, training=False):
        x = self.dense1(z_input)
        x = self.dense2(x)
        return self.dense3(x)


class DiscriminatorXZ(keras.Model):
    def __init__(self, do_spectral_norm: bool):
        super().__init__()
        kernel_initializer = GlorotNormal(seed=500)

        if do_spectral_norm:
            # D(x)
            self.Dx_dense = keras.layers.SpectralNormalization(
                Dense(128, kernel_initializer=kernel_initializer))
            self.Dx_batch_norm = keras.layers.SpectralNormalization(
                BatchNormalization())
            self.Dx_leakyRelu = keras.layers.SpectralNormalization(
                LeakyReLU(0.2))

            # D(z)
            self.Dz_dense = keras.layers.SpectralNormalization(
                Dense(128, kernel_initializer=kernel_initializer))
            self.Dz_leakyRelu = keras.layers.SpectralNormalization(
                LeakyReLU(0.2))
            self.Dz_dropout = keras.layers.SpectralNormalization(
                Dropout(0.5))

            # D(x,z)
            self.Dxz_dense1 = keras.layers.SpectralNormalization(
                Dense(128, kernel_initializer=kernel_initializer))
            self.Dxz_leakyRelu = keras.layers.SpectralNormalization(
                LeakyReLU(0.2))
            self.Dxz_dropout1 = keras.layers.SpectralNormalization(
                Dropout(0.5))
            self.Dxz_dense2 = keras.layers.SpectralNormalization(
                Dense(1, kernel_initializer=kernel_initializer))

        else:
            # D(x)
            self.Dx_dense = Dense(128, kernel_initializer=kernel_initializer)
            self.Dx_batch_norm = BatchNormalization()
            self.Dx_leakyRelu = LeakyReLU(0.2)

            # D(z)
            self.Dz_dense = Dense(128, kernel_initializer=kernel_initializer)
            self.Dz_leakyRelu = LeakyReLU(0.2)
            self.Dz_dropout = Dropout(0.5)

            # D(x,z)
            self.Dxz_dense1 = Dense(128, kernel_initializer=kernel_initializer)
            self.Dxz_leakyRelu = LeakyReLU(0.2)
            self.Dxz_dropout1 = Dropout(0.5)
            self.Dxz_dense2 = Dense(1, kernel_initializer=kernel_initializer)

    def call(self, x_input, z_input, training=False):
        # D(x)
        x = self.Dx_dense(x_input)
        x = self.Dx_batch_norm(x, training=training)
        x = self.Dx_leakyRelu(x)

        # D(z)
        z = self.Dz_dense(z_input)
        z = self.Dz_leakyRelu(z)
        z = self.Dz_dropout(z, training=training)

        # D(x,z)
        y = tf.concat([x, z], axis=1)

        y = self.Dxz_dense1(y)
        y = self.Dxz_leakyRelu(y)
        y = self.Dxz_dropout1(y, training=training)

        y_intermediate = y
        logits = self.Dxz_dense2(y)

        return logits, y_intermediate


class DiscriminatorXX(keras.Model):
    def __init__(self, do_spectral_norm: bool):
        super().__init__()
        kernel_initializer = GlorotNormal(seed=500)

        if do_spectral_norm:
            self.dense1 = keras.layers.SpectralNormalization(
                Dense(128, kernel_initializer=kernel_initializer))
            self.leaky_relu1 = keras.layers.SpectralNormalization(
                LeakyReLU(0.2))
            self.dropout1 = keras.layers.SpectralNormalization(
                Dropout(0.2))
            self.dense2 = keras.layers.SpectralNormalization(
                Dense(1, kernel_initializer=kernel_initializer))
        else:
            self.dense1 = Dense(128, kernel_initializer=kernel_initializer)
            self.leaky_relu1 = LeakyReLU(0.2)
            self.dropout1 = Dropout(0.2)
            self.dense2 = Dense(1, kernel_initializer=kernel_initializer)

    def call(self, x, rec_x, training=False):
        res = tf.concat([x, rec_x], axis=1)

        res = self.dense1(res)
        res = self.leaky_relu1(res)
        res = self.dropout1(res, training=training)

        res_intermediate = res
        logits = self.dense2(res)

        return logits, res_intermediate


class DiscriminatorZZ(keras.Model):
    def __init__(self, do_spectral_norm: bool):
        super().__init__()
        kernel_initializer = GlorotNormal(seed=500)

        if do_spectral_norm:
            self.dense1 = keras.layers.SpectralNormalization(
                Dense(32, kernel_initializer=kernel_initializer))
            self.leaky_relu1 = keras.layers.SpectralNormalization(
                LeakyReLU(0.2))
            self.dropout1 = keras.layers.SpectralNormalization(
                Dropout(0.2))
            self.dense2 = keras.layers.SpectralNormalization(
                Dense(1, kernel_initializer=kernel_initializer))
        else:
            self.dense1 = Dense(32, kernel_initializer=kernel_initializer)
            self.leaky_relu1 = LeakyReLU(0.2)
            self.dropout1 = Dropout(0.2)
            self.dense2 = Dense(1, kernel_initializer=kernel_initializer)

    def call(self, z, rec_z, training=False):
        res = tf.concat([z, rec_z], axis=1)

        res = self.dense1(res)
        res = self.leaky_relu1(res)
        res = self.dropout1(res, training=training)

        res_intermediate = res
        logits = self.dense2(res)

        return logits, res_intermediate