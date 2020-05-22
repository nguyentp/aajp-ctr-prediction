from collections import namedtuple
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


KerasFeat = namedtuple('Feat', ['name', 'nunique', 'Input'])

def make_sum_layer(name='Sum'):
    layer = tf.keras.layers.Lambda(lambda x: K.sum(x, axis=1))
    layer._name = layer.name if name is None else name
    return layer


def make_square_layer(name='Square'):
    layer = tf.keras.layers.Lambda(lambda x: K.square(x))
    layer._name = layer.name if name is None else name
    return layer


def make_half_layer(name='Half'):
    layer = tf.keras.layers.Lambda(lambda x: x*0.5)
    layer._name = layer.name if name is None else name
    return layer


def plot_keras_model(model, save_to):
    return tf.keras.utils.plot_model(model, save_to, show_shapes=True)


def make_keras_feat(name, nunique):
    return KerasFeat(name,
                     nunique,
                     tf.keras.layers.Input(shape=(1), name=name))


def make_feats(num_names, cat_names, cat_nuniques):
    num_feats = [make_keras_feat(name, 1) for name in num_names]
    cat_feats = [make_keras_feat(name, n + 1) for name, n in zip(cat_names, cat_nuniques)]
    return num_feats, cat_feats


def extract_inputs_from_feats(num_feats, cat_feats):
    num_inputs = [f.Input for f in num_feats]
    cat_inputs = [f.Input for f in cat_feats]
    return num_inputs + cat_inputs


def make_embeding(num_feats=None, cat_feats=None, embsz=None):
    xnum = [tf.keras.layers.Dense(embsz)(f.Input) for f in num_feats]
    xnum = [tf.keras.layers.Reshape((1, embsz))(x) for x in xnum]
    xemb = [tf.keras.layers.Embedding(f.nunique, embsz, input_length=1)(f.Input) for f in cat_feats]
    return tf.keras.layers.Concatenate(axis=1)(xnum + xemb)


def make_linear_from_embeding(emb):
    x = tf.keras.layers.Flatten()(emb)
    return tf.keras.layers.Dense(1, name='Linear')(x)


def make_interaction_from_embeding(emb):
    sum_emb = make_sum_layer('sum_emb')(emb)
    square_emb = make_square_layer('square_emb')(emb)
    square_of_sum = tf.keras.layers.Multiply(name='square_of_sum')([sum_emb, sum_emb])
    sum_of_square = make_sum_layer('sum_of_square')(square_emb)
    x = tf.keras.layers.Subtract(name='square_of_sum-sum_of_square')([square_of_sum, sum_of_square])
    x = make_sum_layer()(x)
    x = make_half_layer()(x)
    return tf.keras.layers.Reshape((1,))(x)


def make_fm_model(num_names, cat_names, cat_nuniques, embsz):
    num_feats, cat_feats = make_feats(num_names, cat_names, cat_nuniques)
    
    xemb = make_embeding(num_feats, cat_feats, embsz)
    xlinear = make_linear_from_embeding(xemb)
    xinteraction = make_interaction_from_embeding(xemb)
    
    inputs = extract_inputs_from_feats(num_feats, cat_feats)
    output = tf.keras.layers.Add()([xlinear, xinteraction])
    return tf.keras.models.Model(inputs, output)

        