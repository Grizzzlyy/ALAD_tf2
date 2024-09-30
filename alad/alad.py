import random
import time
import importlib
import os

import tensorflow as tf
import numpy as np
from sklearn import metrics
from tensorflow.keras.utils import plot_model

from alad.eval import score_ch, score_fm, score_l1, score_l2
from alad.utils import batch_fill, create_results_dir


# TODO add display_parameters
# TODO add tqdm
# TODO early stopping
# TODO try glorot_uniform (default) instead of GlorotNormal
# TODO logger
# TODO checkpoint
# TODO batch_fill looks dirty


# TODO move allow_zz to train section


class ALAD:
    def __init__(self, dataset: str, random_seed: int, allow_zz: bool, batch_size: int):
        """
        Parameters
        ----------
        dataset : str
            Name of dataset, on which ALAD will be trained on
        random_seed : int
            For batch norms and dropouts
        allow_zz : bool
            Allow the d_zz discriminator or not for ablation study
        batch_size : int
            Required to build models
        """
        models = importlib.import_module(f"alad.{dataset}_utils")

        # Hyperparameters
        self.real_dim = models.REAL_DIM
        self.latent_dim = models.LATENT_DIM
        self.allow_zz = allow_zz
        self.ema_decay = 0.999
        self.batch_size = batch_size

        # Create results directory
        self.results_dir = create_results_dir(dataset=dataset, allow_zz=allow_zz, random_seed=random_seed)

        # Models
        self.gen = models.Generator(random_seed)
        self.enc = models.Encoder(random_seed)
        self.dis_xz = models.DiscriminatorXZ(random_seed)
        self.dis_xx = models.DiscriminatorXX(random_seed)
        self.dis_zz = models.DiscriminatorZZ(random_seed)

        # Build models
        self.gen.build(batch_size=batch_size)
        self.enc.build(batch_size=batch_size)
        self.dis_xz.build(batch_size=batch_size)
        self.dis_xx.build(batch_size=batch_size)
        self.dis_zz.build(batch_size=batch_size)

        # Optimizers
        self.optimizer_gen = None
        self.optimizer_enc = None
        self.optimizer_dis_xz = None
        self.optimizer_dis_xx = None
        self.optimizer_dis_zz = None

        # ExponentialMovingAverage (EMA)
        self.gen_ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)
        self.enc_ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)
        self.xz_ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)
        self.xx_ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)
        self.zz_ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)

        # Apply EMA after building models
        self.gen_ema.apply([var.value for var in self.gen.trainable_variables])
        self.enc_ema.apply([var.value for var in self.enc.trainable_variables])
        self.xz_ema.apply([var.value for var in self.dis_xz.trainable_variables])
        self.xx_ema.apply([var.value for var in self.dis_xx.trainable_variables])
        self.zz_ema.apply([var.value for var in self.dis_zz.trainable_variables])

        # Temporary buffer for saving actual weights when swapping them to EMA average
        self.vars_tmp = {}

    def train(self, trainx, trainy, epochs: int, batch_size: int, learning_rate: float):
        # Define optimizers
        self.optimizer_gen = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
        self.optimizer_enc = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
        self.optimizer_dis_xz = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
        self.optimizer_dis_xx = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
        self.optimizer_dis_zz = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

        # Shuffle dataset
        np.random.shuffle(trainx)

        # TODO last batch here is not processed i guess
        n_batches = int(trainx.shape[0] / batch_size)

        for epoch in range(epochs):
            # Epoch losses
            train_loss_dis_xz, train_loss_dis_xx, train_loss_dis_zz, train_loss_dis, \
                train_loss_gen, train_loss_enc = [0, 0, 0, 0, 0, 0]
            start_time = time.time()
            for step in range(n_batches):
                # TODO remove this
                if step % 1000 == 0:
                    print(f"step {step}/{n_batches}")
                idx_from = step * batch_size
                idx_to = (step + 1) * batch_size

                x_batch = trainx[idx_from:idx_to]
                z_batch = np.random.normal(size=[batch_size, self.latent_dim]).astype(np.float32)

                ld, ldxz, ldxx, ldzz, le, lg = self.train_step(x_batch, z_batch)

                train_loss_dis += ld
                train_loss_dis_xz += ldxz
                train_loss_dis_xx += ldxx
                train_loss_dis_zz += ldzz
                train_loss_gen += lg
                train_loss_enc += le

            train_loss_gen /= n_batches
            train_loss_enc /= n_batches
            train_loss_dis /= n_batches
            train_loss_dis_xz /= n_batches
            train_loss_dis_xx /= n_batches
            train_loss_dis_zz /= n_batches

            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1} | time = {epoch_time}s | loss gen = {train_loss_gen:.4f} | "
                  f"loss enc = {train_loss_enc:.4f} | loss dis = {train_loss_dis:.4f} | "
                  f"loss dis xz = {train_loss_dis_xz:.4f} | loss dis xx = {train_loss_dis_xx:.4f} |")

    # TODO add this later
    @tf.function
    def train_step(self, x_pl, z_pl):

        ### Train discriminators ###
        with tf.GradientTape(persistent=True) as tape:
            z_gen = self.enc(x_pl, training=True)
            x_gen = self.gen(z_pl, training=True)
            rec_x = self.gen(z_gen, training=True)
            rec_z = self.enc(x_gen, training=True)

            # D(x, z)
            l_encoder, inter_layer_inp_xz = self.dis_xz(x_pl, z_gen, training=True)
            l_generator, inter_layer_rct_xz = self.dis_xz(x_gen, z_pl, training=True)

            # D(x, x)
            x_logit_real, inter_layer_inp_xx = self.dis_xx(x_pl, x_pl, training=True)
            x_logit_fake, inter_layer_rct_xx = self.dis_xx(x_pl, rec_x, training=True)

            # D(z, z)
            z_logit_real, _ = self.dis_zz(z_pl, z_pl, training=True)
            z_logit_fake, _ = self.dis_zz(z_pl, rec_z, training=True)

            ### LOSSES ###

            # D(x,z)
            loss_dis_enc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(l_encoder), logits=l_encoder))
            loss_dis_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(l_generator), logits=l_generator))
            dis_loss_xz = loss_dis_gen + loss_dis_enc

            # D(x,x)
            x_real_dis = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_logit_real, labels=tf.ones_like(x_logit_real))
            x_fake_dis = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_logit_fake, labels=tf.zeros_like(x_logit_fake))
            dis_loss_xx = tf.reduce_mean(x_real_dis + x_fake_dis)

            # D(z,z)
            z_real_dis = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=z_logit_real, labels=tf.ones_like(z_logit_real))
            z_fake_dis = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=z_logit_fake, labels=tf.zeros_like(z_logit_fake))
            dis_loss_zz = tf.reduce_mean(z_real_dis + z_fake_dis)

            loss_discriminator = dis_loss_xz + dis_loss_xx + dis_loss_zz if self.allow_zz else dis_loss_xz + dis_loss_xx

        # Calculate discriminators gradients
        dis_xz_grad = tape.gradient(dis_loss_xz, self.dis_xz.trainable_variables)
        dis_xx_grad = tape.gradient(dis_loss_xx, self.dis_xx.trainable_variables)
        dis_zz_grad = tape.gradient(dis_loss_zz, self.dis_zz.trainable_variables)
        del tape

        # Update discriminators weights
        self.optimizer_dis_xz.apply_gradients(zip(dis_xz_grad, self.dis_xz.trainable_variables))
        self.optimizer_dis_xx.apply_gradients(zip(dis_xx_grad, self.dis_xx.trainable_variables))
        self.optimizer_dis_zz.apply_gradients(zip(dis_zz_grad, self.dis_zz.trainable_variables))

        # Update EMA
        self.xz_ema.apply([var.value for var in self.dis_xz.trainable_variables])
        self.xx_ema.apply([var.value for var in self.dis_xx.trainable_variables])
        self.zz_ema.apply([var.value for var in self.dis_zz.trainable_variables])

        ### Train generator and encoder ###
        with tf.GradientTape(persistent=True) as tape:
            z_gen = self.enc(x_pl, training=True)
            x_gen = self.gen(z_pl, training=True)
            rec_x = self.gen(z_gen, training=True)
            rec_z = self.enc(x_gen, training=True)

            # D(x, z)
            l_encoder, inter_layer_inp_xz = self.dis_xz(x_pl, z_gen, training=True)
            l_generator, inter_layer_rct_xz = self.dis_xz(x_gen, z_pl, training=True)

            # D(x, x)
            x_logit_real, inter_layer_inp_xx = self.dis_xx(x_pl, x_pl, training=True)
            x_logit_fake, inter_layer_rct_xx = self.dis_xx(x_pl, rec_x, training=True)

            # D(z, z)
            z_logit_real, _ = self.dis_zz(z_pl, z_pl, training=True)
            z_logit_fake, _ = self.dis_zz(z_pl, rec_z, training=True)

            ### LOSSES ###

            # D(x,z)
            loss_dis_enc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(l_encoder), logits=l_encoder))
            loss_dis_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(l_generator), logits=l_generator))
            dis_loss_xz = loss_dis_gen + loss_dis_enc

            # D(x,x)
            x_real_dis = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_logit_real, labels=tf.ones_like(x_logit_real))
            x_fake_dis = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_logit_fake, labels=tf.zeros_like(x_logit_fake))
            dis_loss_xx = tf.reduce_mean(x_real_dis + x_fake_dis)

            # D(z,z)
            z_real_dis = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=z_logit_real, labels=tf.ones_like(z_logit_real))
            z_fake_dis = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=z_logit_fake, labels=tf.zeros_like(z_logit_fake))
            dis_loss_zz = tf.reduce_mean(z_real_dis + z_fake_dis)

            loss_discriminator = dis_loss_xz + dis_loss_xx + dis_loss_zz if self.allow_zz else dis_loss_xz + dis_loss_xx

            # G and E losses
            gen_loss_xz = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(l_generator), logits=l_generator))
            enc_loss_xz = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(l_encoder), logits=l_encoder))
            x_real_gen = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_logit_real, labels=tf.zeros_like(x_logit_real))
            x_fake_gen = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_logit_fake, labels=tf.ones_like(x_logit_fake))
            z_real_gen = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=z_logit_real, labels=tf.zeros_like(z_logit_real))
            z_fake_gen = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=z_logit_fake, labels=tf.ones_like(z_logit_fake))

            cost_x = tf.reduce_mean(x_real_gen + x_fake_gen)
            cost_z = tf.reduce_mean(z_real_gen + z_fake_gen)

            cycle_consistency_loss = cost_x + cost_z if self.allow_zz else cost_x
            loss_generator = gen_loss_xz + cycle_consistency_loss
            loss_encoder = enc_loss_xz + cycle_consistency_loss

        # Calculate encoder and generator gradients
        gen_grad = tape.gradient(loss_generator, self.gen.trainable_variables)
        enc_grad = tape.gradient(loss_encoder, self.enc.trainable_variables)
        del tape

        # Update gen and enc weights
        self.optimizer_gen.apply_gradients(zip(gen_grad, self.gen.trainable_variables))
        self.optimizer_enc.apply_gradients(zip(enc_grad, self.enc.trainable_variables))

        # Update EMA
        self.gen_ema.apply([var.value for var in self.gen.trainable_variables])
        self.enc_ema.apply([var.value for var in self.enc.trainable_variables])

        ld, ldxz, ldxx, ldzz = loss_discriminator, dis_loss_xz, dis_loss_xx, dis_loss_zz
        le, lg = loss_encoder, loss_generator

        return ld, ldxz, ldxx, ldzz, le, lg

    def swap_vars(self):
        # Swap actual weights and EMA weights or, if it was swapped, swap back
        if len(self.vars_tmp) == 0:
            # Save variables
            self.vars_tmp["gen"] = self.gen.get_weights()
            self.vars_tmp["enc"] = self.enc.get_weights()
            self.vars_tmp["dis_xz"] = self.dis_xz.get_weights()
            self.vars_tmp["dis_xx"] = self.dis_xx.get_weights()
            self.vars_tmp["dis_zz"] = self.dis_zz.get_weights()

            # Load EMA averages
            for var in self.gen.trainable_variables:
                var.value.assign(self.gen_ema.average(var.value))
            for var in self.enc.trainable_variables:
                var.value.assign(self.enc_ema.average(var.value))
            for var in self.dis_xz.trainable_variables:
                var.value.assign(self.xz_ema.average(var.value))
            for var in self.dis_xx.trainable_variables:
                var.value.assign(self.xx_ema.average(var.value))
            for var in self.dis_zz.trainable_variables:
                var.value.assign(self.zz_ema.average(var.value))
        else:
            # Load saved variables
            self.gen.set_weights(self.vars_tmp["gen"])
            self.enc.set_weights(self.vars_tmp["enc"])
            self.dis_xz.set_weights(self.vars_tmp["dis_xz"])
            self.dis_xx.set_weights(self.vars_tmp["dis_xx"])
            self.dis_zz.set_weights(self.vars_tmp["dis_zz"])

            # Delete them from buffer
            self.vars_tmp = {}

    def test(self, testx, testy, batch_size, degree):
        # TODO change lists to numpy arrays
        scores_ch = []
        scores_l1 = []
        scores_l2 = []
        scores_fm = []
        inference_time = []

        nr_batches_test = int(testx.shape[0] / batch_size)

        # Load EMA weights
        self.swap_vars()

        for t in range(nr_batches_test):
            ran_from = t * batch_size
            ran_to = (t + 1) * batch_size
            x_pl = testx[ran_from:ran_to]
            z_pl = np.random.normal(size=[batch_size, self.latent_dim]).astype(np.float32)

            _score_ch, _score_l1, _score_l2, _score_fm = self.test_step(x_pl, z_pl, degree)

            _score_ch = np.array(_score_ch).tolist()
            _score_l1 = np.array(_score_l1).tolist()
            _score_l2 = np.array(_score_l2).tolist()
            _score_fm = np.array(_score_fm).tolist()

            scores_ch += _score_ch
            scores_l1 += _score_l1
            scores_l2 += _score_l2
            scores_fm += _score_fm

        if testx.shape[0] % batch_size != 0:
            x_pl, size = batch_fill(testx, batch_size)
            z_pl = np.random.normal(size=[batch_size, self.latent_dim]).astype(np.float32)

            _bscore_ch, _bscore_l1, _bscore_l2, _bscore_fm = self.test_step(x_pl, z_pl, degree)

            _bscore_ch = np.array(_bscore_ch).tolist()
            _bscore_l1 = np.array(_bscore_l1).tolist()
            _bscore_l2 = np.array(_bscore_l2).tolist()
            _bscore_fm = np.array(_bscore_fm).tolist()

            scores_ch += _bscore_ch[:size]
            scores_l1 += _bscore_l1[:size]
            scores_l2 += _bscore_l2[:size]
            scores_fm += _bscore_fm[:size]

        # Load regular weights
        self.swap_vars()

        scores_ch = np.array(scores_ch)
        fpr, tpr, _ = metrics.roc_curve(testy, scores_ch)
        roc_auc = metrics.auc(fpr, tpr)
        per = np.percentile(scores_ch, 80)
        y_pred = (scores_ch >= per)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(testy.astype(int),
                                                                           y_pred.astype(int),
                                                                           average='binary')
        print("CH:")
        print(f"roc_auc: {roc_auc}")
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1: {f1}")

        scores_l1 = np.array(scores_l1)
        fpr, tpr, _ = metrics.roc_curve(testy, scores_l1)
        roc_auc = metrics.auc(fpr, tpr)
        per = np.percentile(scores_l1, 80)
        y_pred = (scores_l1 >= per)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(testy.astype(int),
                                                                           y_pred.astype(int),
                                                                           average='binary')
        print("L1:")
        print(f"roc_auc: {roc_auc}")
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1: {f1}")

        scores_l2 = np.array(scores_l2)
        fpr, tpr, _ = metrics.roc_curve(testy, scores_l2)
        roc_auc = metrics.auc(fpr, tpr)
        per = np.percentile(scores_l2, 80)
        y_pred = (scores_l2 >= per)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(testy.astype(int),
                                                                           y_pred.astype(int),
                                                                           average='binary')
        print("L2:")
        print(f"roc_auc: {roc_auc}")
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1: {f1}")

        scores_fm = np.array(scores_fm)
        fpr, tpr, _ = metrics.roc_curve(testy, scores_fm)
        roc_auc = metrics.auc(fpr, tpr)
        per = np.percentile(scores_fm, 80)
        y_pred = (scores_fm >= per)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(testy.astype(int),
                                                                           y_pred.astype(int),
                                                                           average='binary')
        print("FM:")
        print(f"roc_auc: {roc_auc}")
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1: {f1}")

        return scores_ch  # TODO

    # TODO add
    @tf.function
    def test_step(self, x_pl, z_pl, degree):
        z_gen_ema = self.enc(x_pl, training=False)
        rec_x_ema = self.gen(z_gen_ema, training=False)
        x_gen_ema = self.gen(z_pl, training=False)

        l_encoder_emaxx, inter_layer_inp_emaxx = self.dis_xx(x_pl, x_pl, training=False)
        l_generator_emaxx, inter_layer_rct_emaxx = self.dis_xx(x_pl, rec_x_ema, training=False)

        _score_ch = score_ch(l_generator_emaxx)
        _score_l1 = score_l1(x_pl, rec_x_ema)
        _score_l2 = score_l2(x_pl, rec_x_ema)
        _score_fm = score_fm(inter_layer_inp_emaxx, inter_layer_rct_emaxx, degree)
        return _score_ch, _score_l1, _score_l2, _score_fm


def run(args):
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    # Load dataset
    dataset = importlib.import_module(f"data.{args.dataset}.{args.dataset}")
    trainx, trainy = dataset.get_train()
    testx, testy = dataset.get_test()

    # TODO remove batch_size from train and test
    alad = ALAD(dataset=args.dataset, random_seed=args.seed, allow_zz=args.enable_dzz, batch_size=50)

    alad.train(trainx, trainy, args.epochs, args.batch_size, learning_rate=1e-5)

    alad.test(testx, testy, args.batch_size, args.degree)
