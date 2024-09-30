import random
import sys
import time
import importlib
import os

import tensorflow as tf
import numpy as np
from sklearn import metrics
# from tensorflow.keras.utils import plot_model
from tqdm import tqdm
import pandas as pd

from alad.eval import score_ch, score_fm, score_l1, score_l2
from alad.utils import batch_fill, create_results_dir, create_logger, print_parameters


# TODO add display_parameters
# TODO early stopping
# TODO try glorot_uniform (default) instead of GlorotNormal
# TODO checkpoint
# TODO batch_fill looks dirty
# TODO continue training


class ALAD:
    def __init__(self, dataset_name: str, random_seed: int, allow_zz: bool, batch_size: int):
        """
        Parameters
        ----------
        dataset_name : str
            Name of dataset, on which ALAD will be trained on
        random_seed : int
            For batch norms and dropouts
        allow_zz : bool
            Allow the d_zz discriminator or not for ablation study
        batch_size : int
            Required to build models
        """

        # Module with encoder, generator, discriminators and some hyperparameters for this dataset
        models_module = importlib.import_module(f"alad.{dataset_name}_utils")

        # Hyperparameters
        self.real_dim = models_module.REAL_DIM
        self.latent_dim = models_module.LATENT_DIM
        self.allow_zz = allow_zz
        self.ema_decay = 0.999
        self.batch_size = batch_size
        self.learning_rate = models_module.LEARNING_RATE

        # Create models
        self.models = {"gen": {"model": models_module.Generator(random_seed)},
                       "enc": {"model": models_module.Encoder(random_seed)},
                       "dis_xz": {"model": models_module.DiscriminatorXZ(random_seed)},
                       "dis_xx": {"model": models_module.DiscriminatorXX(random_seed)},
                       "dis_zz": {"model": models_module.DiscriminatorZZ(random_seed)}}

        # Build models
        for model_dict in self.models.values():
            model_dict["model"].build(batch_size=batch_size)

        # Optimizers
        for model_dict in self.models.values():
            model_dict["optimizer"] = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5)

        # ExponentialMovingAverage (EMA)
        for model_dict in self.models.values():
            model_dict["ema"] = tf.train.ExponentialMovingAverage(decay=self.ema_decay)

        # Apply EMA for the first time after building models
        for model_dict in self.models.values():
            model_dict["ema"].apply([var.value for var in model_dict["model"].trainable_variables])

        # Create results directory
        self.results_dir = create_results_dir(dataset=dataset_name, allow_zz=allow_zz, random_seed=random_seed)
        # Create logger
        self.logger = create_logger(dataset_name, allow_zz, random_seed)

        # Display parameters
        print_parameters(self.logger, **{"dataset": dataset_name, "allow_zz": allow_zz, "random_seed": random_seed,
                                         "batch_size": batch_size})

        # Loss history
        self.loss_hist = {"epoch": [], "gen": [], "enc": [], "dis": [], "dis_xz": [], "dis_xx": [], "dis_zz": []}

        self.epochs_done = 0
        self.is_ema_swapped = False  # are actual weights swapped with EMA weigts

    def swap_vars(self):
        """
        Swap actual weights and EMA weights or, if it was swapped, swap back
        """
        if self.is_ema_swapped:
            # Load saved variables
            for model_name in self.models:
                self.models[model_name]["model"].set_weights(
                    self.models[model_name]["vars_tmp"]
                )
                del self.models[model_name]["vars_tmp"]
            self.is_ema_swapped = False
        else:
            # TODO here is
            # Save weights
            for model_name in self.models:
                self.models[model_name]["vars_tmp"] = self.models[model_name]["model"].get_weights()
            # Swap to EMA weights
            for model_name in self.models:
                for var in self.models[model_name]["model"].trainable_variables:
                    var.value.assign(self.models[model_name]["ema"].average(var.value))
            self.is_ema_swapped = True

    def train(self, trainx, epochs: int):
        self.logger.info(f"Training epochs {0} - {epochs - 1}")

        # Shuffle dataset
        np.random.shuffle(trainx)

        # TODO last batch here is not processed i guess
        n_batches = int(trainx.shape[0] / self.batch_size)

        for epoch in range(epochs):
            # Epoch losses
            epoch_losses = {"gen": 0, "enc": 0, "dis": 0, "dis_xz": 0, "dis_xx": 0, "dis_zz": 0}
            # train_loss_dis_xz, train_loss_dis_xx, train_loss_dis_zz, train_loss_dis, \
            #     train_loss_gen, train_loss_enc = [0, 0, 0, 0, 0, 0]
            start_time = time.time()

            for step in tqdm(range(n_batches), desc=f"Epoch {epoch}", file=sys.stdout):
                idx_from = step * self.batch_size
                idx_to = (step + 1) * self.batch_size

                x_batch = trainx[idx_from:idx_to]
                z_batch = np.random.normal(size=[self.batch_size, self.latent_dim]).astype(np.float32)

                ld, ldxz, ldxx, ldzz, le, lg = self.train_step(x_batch, z_batch)

                epoch_losses["gen"] += lg
                epoch_losses["enc"] += le
                epoch_losses["dis"] += ld
                epoch_losses["dis_xz"] += ldxz
                epoch_losses["dis_xx"] += ldxx
                epoch_losses["dis_zz"] += ldzz

            for key in epoch_losses:
                epoch_losses[key] /= n_batches

            self.loss_hist["epoch"].append(epoch)
            for key in epoch_losses:
                self.loss_hist[key].append(tf.get_static_value(epoch_losses[key]))

            epoch_time = time.time() - start_time
            if self.allow_zz:
                self.logger.info(
                    f"epoch {epoch} | time = {epoch_time:.2f}s | loss gen = {epoch_losses['gen']:.4f} | "
                    f"loss enc = {epoch_losses['enc']:.4f} | loss dis = {epoch_losses['dis']:.4f} | "
                    f"loss dis xz = {epoch_losses['dis_xz']:.4f} | loss dis xx = {epoch_losses['dis_xx']:.4f} | "
                    f"loss dis zz = {epoch_losses['dis_zz']:.4f} |")
            else:
                self.logger.info(
                    f"epoch {epoch} | time = {epoch_time:.2f}s | loss gen = {epoch_losses['gen']:.4f} | "
                    f"loss enc = {epoch_losses['enc']:.4f} | loss dis = {epoch_losses['dis']:.4f} | "
                    f"loss dis xz = {epoch_losses['dis_xz']:.4f} | loss dis xx = {epoch_losses['dis_xx']:.4f} |")

            self.epochs_done += 1

        # Save losses to csv
        losses_df = pd.DataFrame(self.loss_hist)
        losses_df.to_csv(os.path.join(self.results_dir, "train_loss.csv"), index=False)

    # TODO add this later
    @tf.function
    def train_step(self, x_pl, z_pl):

        ### Train discriminators ###
        with tf.GradientTape(persistent=True) as tape:
            z_gen = self.models["enc"]["model"](x_pl, training=True)
            x_gen = self.models["gen"]["model"](z_pl, training=True)
            rec_x = self.models["gen"]["model"](z_gen, training=True)
            rec_z = self.models["enc"]["model"](x_gen, training=True)

            # D(x, z)
            l_encoder, inter_layer_inp_xz = self.models["dis_xz"]["model"](x_pl, z_gen, training=True)
            l_generator, inter_layer_rct_xz = self.models["dis_xz"]["model"](x_gen, z_pl, training=True)

            # D(x, x)
            x_logit_real, inter_layer_inp_xx = self.models["dis_xx"]["model"](x_pl, x_pl, training=True)
            x_logit_fake, inter_layer_rct_xx = self.models["dis_xx"]["model"](x_pl, rec_x, training=True)

            # D(z, z)
            z_logit_real, _ = self.models["dis_zz"]["model"](z_pl, z_pl, training=True)
            z_logit_fake, _ = self.models["dis_zz"]["model"](z_pl, rec_z, training=True)

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

        ld, ldxz, ldxx, ldzz = loss_discriminator, dis_loss_xz, dis_loss_xx, dis_loss_zz

        # Calculate discriminators gradients
        dis_xz_grad = tape.gradient(dis_loss_xz, self.models["dis_xz"]["model"].trainable_variables)
        dis_xx_grad = tape.gradient(dis_loss_xx, self.models["dis_xx"]["model"].trainable_variables)
        dis_zz_grad = tape.gradient(dis_loss_zz, self.models["dis_zz"]["model"].trainable_variables)
        del tape

        # Update discriminators weights
        self.models["dis_xz"]["optimizer"].apply_gradients(
            zip(dis_xz_grad, self.models["dis_xz"]["model"].trainable_variables))
        self.models["dis_xx"]["optimizer"].apply_gradients(
            zip(dis_xx_grad, self.models["dis_xx"]["model"].trainable_variables))
        self.models["dis_zz"]["optimizer"].apply_gradients(
            zip(dis_zz_grad, self.models["dis_zz"]["model"].trainable_variables))

        # Update EMA
        discriminators = ["dis_xz", "dis_xx", "dis_zz"]
        for discr in discriminators:
            self.models[discr]["ema"].apply([var.value for var in self.models[discr]["model"].trainable_variables])

        ### Train generator and encoder ###
        with tf.GradientTape(persistent=True) as tape:
            z_gen = self.models["enc"]["model"](x_pl, training=True)
            x_gen = self.models["gen"]["model"](z_pl, training=True)
            rec_x = self.models["gen"]["model"](z_gen, training=True)
            rec_z = self.models["enc"]["model"](x_gen, training=True)

            # D(x, z)
            l_encoder, inter_layer_inp_xz = self.models["dis_xz"]["model"](x_pl, z_gen, training=True)
            l_generator, inter_layer_rct_xz = self.models["dis_xz"]["model"](x_gen, z_pl, training=True)

            # D(x, x)
            x_logit_real, inter_layer_inp_xx = self.models["dis_xx"]["model"](x_pl, x_pl, training=True)
            x_logit_fake, inter_layer_rct_xx = self.models["dis_xx"]["model"](x_pl, rec_x, training=True)

            # D(z, z)
            z_logit_real, _ = self.models["dis_zz"]["model"](z_pl, z_pl, training=True)
            z_logit_fake, _ = self.models["dis_zz"]["model"](z_pl, rec_z, training=True)

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

        le, lg = loss_encoder, loss_generator

        # Calculate encoder and generator gradients
        gen_grad = tape.gradient(loss_generator, self.models["gen"]["model"].trainable_variables)
        enc_grad = tape.gradient(loss_encoder, self.models["enc"]["model"].trainable_variables)
        del tape

        # Update gen and enc weights
        self.models["gen"]["optimizer"].apply_gradients(zip(gen_grad, self.models["gen"]["model"].trainable_variables))
        self.models["enc"]["optimizer"].apply_gradients(zip(enc_grad, self.models["enc"]["model"].trainable_variables))

        # Update EMA
        self.models["gen"]["ema"].apply([var.value for var in self.models["gen"]["model"].trainable_variables])
        self.models["enc"]["ema"].apply([var.value for var in self.models["enc"]["model"].trainable_variables])

        return ld, ldxz, ldxx, ldzz, le, lg

    def test(self, testx, testy, degree):
        # TODO change lists to numpy arrays
        scores_ch = []
        scores_l1 = []
        scores_l2 = []
        scores_fm = []
        inference_time = []

        nr_batches_test = int(testx.shape[0] / self.batch_size)

        # Load EMA weights
        self.swap_vars()

        for t in tqdm(range(nr_batches_test), desc="Evaluating", file=sys.stdout):
            ran_from = t * self.batch_size
            ran_to = (t + 1) * self.batch_size
            x_pl = testx[ran_from:ran_to]
            z_pl = np.random.normal(size=[self.batch_size, self.latent_dim]).astype(np.float32)
            begin_test_time_batch = time.time()

            _score_ch, _score_l1, _score_l2, _score_fm = self.test_step(x_pl, z_pl, degree)

            scores_ch += np.array(_score_ch).tolist()
            scores_l1 += np.array(_score_l1).tolist()
            scores_l2 += np.array(_score_l2).tolist()
            scores_fm += np.array(_score_fm).tolist()
            inference_time.append(time.time() - begin_test_time_batch)

        # Process last batch
        if testx.shape[0] % self.batch_size != 0:
            # TODO change batch_fill
            x_pl, size = batch_fill(testx, self.batch_size)
            z_pl = np.random.normal(size=[self.batch_size, self.latent_dim]).astype(np.float32)
            begin_test_time_batch = time.time()

            _bscore_ch, _bscore_l1, _bscore_l2, _bscore_fm = self.test_step(x_pl, z_pl, degree)

            _bscore_ch = np.array(_bscore_ch).tolist()
            _bscore_l1 = np.array(_bscore_l1).tolist()
            _bscore_l2 = np.array(_bscore_l2).tolist()
            _bscore_fm = np.array(_bscore_fm).tolist()

            scores_ch += _bscore_ch[:size]
            scores_l1 += _bscore_l1[:size]
            scores_l2 += _bscore_l2[:size]
            scores_fm += _bscore_fm[:size]
            inference_time.append(time.time() - begin_test_time_batch)

        # Load regular weights
        self.swap_vars()

        # Print inference time
        inference_time = np.mean(inference_time)
        self.logger.info(f"Mean inference time: {inference_time}")

        # print(f"Mean inference time: {inference_time}")

        def get_metrics(scores: list):
            scores = np.array(scores)
            fpr, tpr, _ = metrics.roc_curve(testy, scores)
            roc_auc = metrics.auc(fpr, tpr)
            # TODO you can change percentile here
            per = np.percentile(scores, 80)
            y_pred = (scores >= per)
            precision, recall, f1, _ = metrics.precision_recall_fscore_support(testy.astype(int),
                                                                               y_pred.astype(int),
                                                                               average='binary')

            return {"roc_auc": roc_auc, "precision": precision, "recall": recall, "f1": f1}

        def print_metrics(name: str, **kwargs):
            # name - metric name, **kwargs - dict with metrics
            self.logger.info(f"{name}:")
            for name, value in kwargs.items():
                self.logger.info(f"{name}: {value}")

        print_metrics(name="CH", **get_metrics(scores_ch))
        print_metrics(name="L1", **get_metrics(scores_l1))
        print_metrics(name="L2", **get_metrics(scores_l2))
        print_metrics(name="FM", **get_metrics(scores_fm))

        metrics_ = {"epoch": [self.epochs_done] * 4, "method": ["CH", "L1", "L2", "FM"],
                    "roc_auc": [], "precision": [], "recall": [], "f1": []}
        for scores in [scores_ch, scores_l1, scores_l2, scores_fm]:
            scores_metrics = get_metrics(scores)
            metrics_["roc_auc"].append(scores_metrics["roc_auc"])
            metrics_["precision"].append(scores_metrics["precision"])
            metrics_["recall"].append(scores_metrics["recall"])
            metrics_["f1"].append(scores_metrics["f1"])

        metrics_df = pd.DataFrame(metrics_)
        metrics_df.to_csv(os.path.join(self.results_dir, "metrics.csv"), index=False)

    # TODO add
    @tf.function
    def test_step(self, x_pl, z_pl, degree):
        z_gen_ema = self.models["enc"]["model"](x_pl, training=False)
        rec_x_ema = self.models["gen"]["model"](z_gen_ema, training=False)
        # x_gen_ema = self.gen(z_pl, training=False)

        l_encoder_emaxx, inter_layer_inp_emaxx = self.models["dis_xx"]["model"](x_pl, x_pl, training=False)
        l_generator_emaxx, inter_layer_rct_emaxx = self.models["dis_xx"]["model"](x_pl, rec_x_ema, training=False)

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

    alad = ALAD(dataset_name=args.dataset_name, random_seed=args.seed, allow_zz=args.enable_dzz,
                batch_size=args.batch_size)

    # Load dataset
    dataset = importlib.import_module(f"data.{args.dataset_name}.{args.dataset_name}")
    trainx, trainy = dataset.get_train()
    testx, testy = dataset.get_test()

    alad.train(trainx, args.epochs)

    alad.test(testx, testy, args.degree)
