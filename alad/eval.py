import tensorflow as tf
from tensorflow.keras.layers import Flatten


# TODO Delete comments
@tf.function
def score_ch(l_generator_emaxx):
    score_ch = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(l_generator_emaxx),
        logits=l_generator_emaxx)
    # score_ch = tf.squeeze(score_ch)
    return score_ch


@tf.function
def score_l1(x_pl, rec_x_ema):
    rec = x_pl - rec_x_ema
    # rec = Flatten()(rec)
    score_l1 = tf.norm(rec, ord=1, axis=1,
                       keepdims=False, name='d_loss')
    # score_l1 = tf.squeeze(score_l1)
    return score_l1


@tf.function
def score_l2(x_pl, rec_x_ema):
    rec = x_pl - rec_x_ema
    # rec = Flatten()(rec)
    score_l2 = tf.norm(rec, ord=2, axis=1,
                       keepdims=False, name='d_loss')
    # score_l2 = tf.squeeze(score_l2)
    return score_l2


@tf.function
def score_fm(inter_layer_inp_emaxx, inter_layer_rct_emaxx, degree):
    inter_layer_inp, inter_layer_rct = inter_layer_inp_emaxx, \
        inter_layer_rct_emaxx
    fm = inter_layer_inp - inter_layer_rct
    # fm = Flatten()(fm)
    score_fm = tf.norm(fm, ord=degree, axis=1,
                       keepdims=False, name='d_loss')
    # score_fm = tf.squeeze(score_fm)
    return score_fm
