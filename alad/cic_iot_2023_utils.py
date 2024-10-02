"""

CIC_IOT_2023 ALAD architecture.

Generator (decoder), encoder and discriminator.

"""
import tensorflow as tf
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Dropout, Concatenate

LEARNING_RATE = 1e-5
REAL_DIM = 63
LATENT_DIM = 32

class Encoder(tf.keras.Layer):
    """
    Encoder architecture in tensorflow. Maps real data into the latent space.
    """

    def __init__(self, random_seed):
        super().__init__()
        kernel_initializer = GlorotNormal(seed=random_seed)

        self.dense1 = Dense(64, kernel_initializer=kernel_initializer)
        self.leaky_relu1 = LeakyReLU(0.2)
        self.dense2 = Dense(LATENT_DIM, kernel_initializer=kernel_initializer)

    def build(self, batch_size):
        # Call on dummy data to initialize weights
        dummy_tensor = tf.zeros((batch_size, REAL_DIM))
        self.call(dummy_tensor, training=False)

    def call(self, x_input, training=False):
        """
        Parameters
        ----------
        x_input : tensor
            Data from real space
        training : bool, optional
            For batch norms and dropouts

        Returns
        -------
        net : tensor
        """
        z = self.dense1(x_input)
        z = self.leaky_relu1(z)
        return self.dense2(z)


class Generator(tf.keras.Layer):
    def __init__(self, random_seed):
        super().__init__()
        kernel_initializer = GlorotNormal(seed=random_seed)

        self.dense1 = Dense(64, activation="relu", kernel_initializer=kernel_initializer)
        self.dense2 = Dense(128, activation="relu", kernel_initializer=kernel_initializer)
        self.dense3 = Dense(REAL_DIM, kernel_initializer=kernel_initializer)

    def build(self, batch_size):
        # Call on dummy data to initialize weights
        dummy_tensor = tf.zeros((batch_size, LATENT_DIM))
        self.call(dummy_tensor, training=False)

    def call(self, z_input, training=False):
        """
        Parameters
        ----------
        z_input : tensor
            Data from latent space
        training : bool, optional
            For batch norms and dropouts

        Returns
        -------
        net : tensor
        """
        x = self.dense1(z_input)
        x = self.dense2(x)
        return self.dense3(x)


class DiscriminatorXZ(tf.keras.Layer):
    def __init__(self, random_seed):
        super().__init__()
        kernel_initializer = GlorotNormal(seed=random_seed)

        # D(x)
        self.Dx_dense = Dense(128, kernel_initializer=kernel_initializer)
        self.Dx_batch_norm = BatchNormalization()
        self.Dx_leakyRelu = LeakyReLU(0.2)

        # D(z)
        self.Dz_dense = Dense(128, kernel_initializer=kernel_initializer)
        self.Dz_leakyRelu = LeakyReLU(0.2)
        self.Dz_dropout = Dropout(0.5)

        # D(x,z)
        self.Dxz_concat = Concatenate(axis=-1)
        self.Dxz_dense1 = Dense(128, kernel_initializer=kernel_initializer)
        self.Dxz_leakyRelu1 = LeakyReLU(0.2)
        self.Dxz_dropout1 = Dropout(0.5)
        self.Dxz_dense2 = Dense(1, kernel_initializer=kernel_initializer)

    def build(self, batch_size):
        # Call on dummy data to initialize weights
        dummy_tensor1 = tf.zeros((batch_size, REAL_DIM))
        dummy_tensor2 = tf.zeros((batch_size, LATENT_DIM))
        self.call(dummy_tensor1, dummy_tensor2, training=False)

    def call(self, x_input, z_input, training=False):
        """
        Parameters
        ----------
        x_input : tensor
            Data from real space
        z_input : tensor
            Data from latent space
        training : bool, optional
            For batch norms and dropouts

        Returns
        -------
        (logits, y_intermediate) : (tensor, tensor)
        """

        # D(x)
        x = self.Dx_dense(x_input)
        x = self.Dx_batch_norm(x, training=training)
        x = self.Dx_leakyRelu(x)

        # D(z)
        z = self.Dz_dense(z_input)
        z = self.Dz_leakyRelu(z)
        z = self.Dz_dropout(z, training=training)

        # D(x,z)
        y = self.Dxz_concat([x, z])
        y = self.Dxz_dense1(y)
        y = self.Dxz_leakyRelu1(y)
        y = self.Dxz_dropout1(y, training=training)

        y_intermediate = y
        logits = self.Dxz_dense2(y)

        return logits, y_intermediate


class DiscriminatorXX(tf.keras.Layer):
    def __init__(self, random_seed):
        super().__init__()
        kernel_initializer = GlorotNormal(seed=random_seed)

        self.concat = Concatenate(axis=-1)
        self.dense1 = Dense(128, kernel_initializer=kernel_initializer)
        self.leaky_relu1 = LeakyReLU(0.2)
        self.dropout1 = Dropout(0.2)
        self.dense2 = Dense(1, kernel_initializer=kernel_initializer)

    def build(self, batch_size):
        # Call on dummy data to initialize weights
        dummy_tensor = tf.zeros((batch_size, REAL_DIM))
        self.call(dummy_tensor, dummy_tensor, training=False)

    def call(self, x, rec_x, training=False):
        """
        Parameters
        ----------
        x : tensor
            Data from real space
        rec_x : tensor
            Recreated data (real space)
        training : bool, optional
            For batch norms and dropouts

        Returns
        -------
        (logits, res_intermediate) : (tensor, tensor)
        """

        res = self.concat([x, rec_x])
        res = self.dense1(res)
        res = self.leaky_relu1(res)
        res = self.dropout1(res, training=training)

        res_intermediate = res
        logits = self.dense2(res)

        return logits, res_intermediate


class DiscriminatorZZ(tf.keras.Layer):
    def __init__(self, random_seed):
        super().__init__()
        kernel_initializer = GlorotNormal(seed=random_seed)

        self.concat = Concatenate(axis=-1)
        self.dense1 = Dense(32, kernel_initializer=kernel_initializer)
        self.leaky_relu1 = LeakyReLU(0.2)
        self.dropout1 = Dropout(0.2)
        self.dense2 = Dense(1, kernel_initializer=kernel_initializer)

    def build(self, batch_size):
        # Call on dummy data to initialize weights
        dummy_tensor = tf.zeros((batch_size, LATENT_DIM))
        self.call(dummy_tensor, dummy_tensor, training=False)

    def call(self, z, rec_z, training=False):
        """
        Parameters
        ----------
        z : tensor
            Data from latent space
        rec_z : tensor
            Recreated data (latent space)
        training : bool, optional
            For batch norms and dropouts

        Returns
        -------
        (logits, net_intermediate) : (tensor, tensor)
        """

        net = self.concat([z, rec_z])
        net = self.dense1(net)
        net = self.leaky_relu1(net)
        net = self.dropout1(net, training=training)

        net_intermediate = net
        logits = self.dense2(net)

        return logits, net_intermediate
