
import time
import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_text
import keras_nlp



# ------------------Model Config------------------

class TransformerConfig:
  _MAX_TOKENS = 128

  def __init__(self, encoderDecoder = False, max_tokens=128, pos_encoding_length=2048, vocab_size = 5000,
               d_model=128, dff=512, num_heads_enc=8, num_heads_dec=8, num_layers_dec=4, num_layers_enc=4, skip_connection_enc=None, skip_connection_dec=None,
               img_width = 256, img_height = 256, hasInpImage = False, hasOutImage = False,

               dropout_rate=0.2, batch_size=32, buffer_size=2000, name="transformer", projectPath=""):

    #model || model size O(vocab_size * d_model)
    self.encoderDecoder = encoderDecoder #if True --> use encoder and decoder model, if False --> only decoder
    self.max_tokens=max_tokens; TransformerConfig._MAX_TOKENS = max_tokens
    self.vocab_size=vocab_size
    self.pos_encoding_length=pos_encoding_length # pos_encoding_length >= max_tokens

    self.d_model=d_model #  == pos_encoding_depth
    self.dff=dff

    self.num_heads_enc=num_heads_enc
    self.num_heads_dec=num_heads_dec
    self.num_layers_dec=num_layers_dec
    self.num_layers_enc=num_layers_enc
    self.skip_connection_enc=skip_connection_enc
    self.skip_connection_dec=skip_connection_dec

    #img hyper param
    self.hasInpImage = hasInpImage
    self.hasOutImage = hasOutImage
    self.hasImage = self.hasInpImage or self.hasOutImage

    self.img_width  = img_width;
    self.img_height = img_height;

    #video hper param
    self.hasInpVideo = False
    self.hasOutVideo = False
    self.hasVideo = self.hasInpVideo or self.hasOutVideo

    self.vid_width  = img_width;
    self.vid_height = img_height;
    # self.vid_nbr_frame or vid len(s)



    self.isMultimodal = self.hasImage or self.hasVideo


    self.dropout_rate=dropout_rate

    #env
    self.batch_size=batch_size
    self.buffer_size=buffer_size
    self.name = name
    self.projectPath = projectPath
    self.checkpoint_path = os.path.join(projectPath,  name)

  def save_config(self, ):
    _fpath = os.path.join(  self.checkpoint_path, "training_config.pkl" )
    with open(_fpath, 'wb') as pfile:
      pickle.dump(self, pfile, protocol=pickle.HIGHEST_PROTOCOL)
    return None




# ------------------Tokenizer------------------

class CustomTokenizer(keras_nlp.models.MistralPreprocessor): # https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/models
      def __init__( self, return_mask: bool = False,  **kwargs,):
        super().__init__(**kwargs)
        self.return_mask = return_mask # no need Mask as Embedding(mask_zero=True) generate mask, make sure padding token id == 0
        self.vocab_size = self.tokenizer.vocabulary_size()

      def call(self, x, y=None, sample_weight=None, sequence_length=None, add_start_token=True, add_end_token=True ):
        self.add_start_token = add_start_token;
        self.add_end_token   = add_end_token;
        x = super().call(x, y=y, sample_weight=sample_weight, sequence_length=sequence_length)
        return x if self.return_mask else x["token_ids"]

      def detokenize(self, x, **kwargs,):
        return self.tokenizer.detokenize(x, **kwargs)







# ----------------------- Positional Embedding ---------------------------

def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1)

  return tf.cast(pos_encoding, dtype=tf.float32)






class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model, pos_encoding_length, name = "pos_embediing"):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model, mask_zero=True, name=name + "_embeding_layer" )
    self.pos_encoding = positional_encoding(length=pos_encoding_length, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x




# --------------------Attention Layers------------------------

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    _layer_name = kwargs.get('name',"attention")
    super().__init__(name = _layer_name  )
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization( name =  _layer_name + "_layer_norm" )
    self.add = tf.keras.layers.Add( name =  _layer_name + "_adding" )

    self.last_attn_scores = None

  def call(self, query, key, value, use_causal_mask=False, mask=None):
    attn_output, attn_scores = self.mha(
        query=query,
        key=key,
        value=value,
        use_causal_mask = use_causal_mask,
        return_attention_scores=True)

    self.last_attn_scores = attn_scores

    x = self.add([query, attn_output])
    x = self.layernorm(x)

    return x;




class CrossAttention(BaseAttention):
  def call(self, x, context, **kwargs ):
    x = super().call( query = x, key = context, value = context, **kwargs  )
    return x



class GlobalSelfAttention(BaseAttention):
  def call(self, x, **kwargs):
    x = super().call( query = x, key = x, value = x, **kwargs  )
    return x



class CausalSelfAttention(BaseAttention):
  def call(self, x, **kwargs):
    x = super().call( query = x, key = x, value = x, use_causal_mask = True, **kwargs   )
    return x



#------------------------ParametricAddLayer---------------------------
#call(x, y) = x + activation(scale) * y <==> x + k * y
#if activation  = tanh ==>  k € [-1, 1]
class ParametricAddLayer(tf.keras.layers.Layer):
    def __init__(self, init_val = 0.5, activation="tanh", name ="parametricAdd", **kwargs):
        super().__init__(**kwargs)
        self.scale = tf.Variable(init_val, trainable=True, dtype=tf.float32, name = "_scaler")
        self.add = tf.keras.layers.Add( name = name + "_add" )
        self.activation = tf.keras.layers.Activation( activation, name = "_activation",  **kwargs)

    def call(self, x, y, ):
        return self.add( [ x,   y * self.activation(self.scale) ] )





# -------------------------------FeedForward-------------------------
class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1, name="feedforward"):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu', name = name + "_dense0"),
      tf.keras.layers.Dense(d_model, name = name + "_dense1"),
      tf.keras.layers.Dropout(dropout_rate, name = name + "_dropout")
    ])
    self.add = tf.keras.layers.Add( name = name + "_add" )
    self.layer_norm = tf.keras.layers.LayerNormalization(name = name + "_layer_norm")

  def call(self, x, ):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x)
    return x



#------------------ ENCODER ------------------------------------#

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1, name="encoder"):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate,
        name= name + "_global_selfAttention")

    self.ffn = FeedForward(d_model = d_model, dff = dff, name = name + "_feedforward")

  def call(self, x, ):
    x = self.self_attention(x=x)
    x = self.ffn(x=x)
    return x





class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, pos_encoding_length, dropout_rate=0.1, skip_connection = None, name="encoder"):
    super(Encoder, self).__init__(name=name)

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(
        vocab_size=vocab_size, d_model=d_model, pos_encoding_length=pos_encoding_length, name= name + "_posEmbedding")

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate,
                     name = name + "_encoderLayer" + str(i))
        for i in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate, name = name + "_dropout")

    #add skip connection
    self.skip_connection = skip_connection
    num_skip_connection  = max(num_layers - skip_connection, 0 ) if skip_connection != None else -1
    self.hasSkip_connection: bool = num_skip_connection > 0
    self.skip_connection_layer = [  ParametricAddLayer(name = name + "_paramSkip" + str(i) ) for i in range(num_skip_connection) ]

  def call(self, x, ):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x=x)  # Shape `(batch_size, seq_len, d_model)`.

    # Add dropout.
    x = self.dropout(x)
    temp_layer = [];
    for i in range(self.num_layers):

      #add skip connection
      if self.hasSkip_connection and i >= self.skip_connection:
        x = self.skip_connection_layer[i - self.skip_connection](x=x, y=temp_layer[ - self.skip_connection ])

      x = self.enc_layers[i](x=x)

      if self.hasSkip_connection:
        temp_layer.append(x)
    return x  # Shape `(batch_size, seq_len, d_model)`.




#------------------ DECODER ------------------------------------#
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1,
               name = "decoder"):
    super().__init__(name=name)

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate,
        name = name + "_causal_selfAttention")

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate,
        name = name + "_causal_crossAttention")

    self.ffn = FeedForward(d_model = d_model, dff = dff, name = name + "_feedforward")

  def call(self, x, context = None, ):
    x = self.causal_self_attention(x=x)
    if context != None:
      x = self.cross_attention(x=x, context=context)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x=x)  # Shape `(batch_size, seq_len, d_model)`.
    return x




class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, pos_encoding_length,
               dropout_rate=0.1, skip_connection = None, name="decoder", **kwargs):
    super().__init__(**kwargs)

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model,
                                             pos_encoding_length=pos_encoding_length, name= name + "_posEmbedding")
    self.dropout = tf.keras.layers.Dropout(dropout_rate, name = name + "_dropout")
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate, name = name + "_decoderLayer" + str(i))
        for i in range(num_layers)]


    #add skip connection
    self.skip_connection = skip_connection
    num_skip_connection  = max(num_layers - skip_connection, 0 ) if skip_connection != None else -1
    self.hasSkip_connection: bool = num_skip_connection > 0
    self.skip_connection_layer = [  ParametricAddLayer(name = name + "_paramSkip" + str(i) ) for i in range(num_skip_connection) ]

    self.last_attn_scores = None

  def call(self, x, context = None, fx_intercept = None,):
    # `x` is token-IDs shape (batch, target_seq_len)
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

    x = self.dropout(x)
    temp_layer = []

    if fx_intercept == None:
      def _fx(x): return x, context
      fx_intercept = _fx

    for i in range(self.num_layers):
      #add skip connection
      if self.hasSkip_connection and i >= self.skip_connection:
        x = self.skip_connection_layer[i - self.skip_connection](x=x, y=temp_layer[ - self.skip_connection ])


      x, context = fx_intercept(x)

      x = self.dec_layers[i](x=x, context = context)

      if self.hasSkip_connection:
        temp_layer.append(x)


    self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    # The shape of x is (batch_size, target_seq_len, d_model).
    return x



# ----------------------------- Transformer ----------------------------------------------------

class Transformer(tf.keras.Model):
  def __init__(self, *, d_model,  dff,
               num_layers_enc, num_layers_dec, num_heads_enc, num_heads_dec, skip_connection_enc, skip_connection_dec,
               input_vocab_size, target_vocab_size, pos_encoding_length, dropout_rate=0.1,  **kwargs):
    super().__init__(**kwargs)

    self.encoder = Encoder(num_layers=num_layers_enc, d_model=d_model,
                           num_heads=num_heads_enc, dff=dff, skip_connection=skip_connection_enc,
                           vocab_size=input_vocab_size,
                           pos_encoding_length=pos_encoding_length,
                           dropout_rate=dropout_rate,
                           name="encoder")

    self.decoder = Decoder(num_layers=num_layers_dec, d_model=d_model,
                           num_heads=num_heads_dec, dff=dff, skip_connection=skip_connection_dec,
                           vocab_size=target_vocab_size,
                           pos_encoding_length=pos_encoding_length,
                           dropout_rate=dropout_rate,
                           name="decoder")

    self.final_layer = tf.keras.layers.Dense(target_vocab_size, name="final_layer")

  def call(self, inputs, fx_intercept=None):

    context, x  = inputs["input_text_encoder"], inputs["input_text_decoder"]


    context = self.encoder(context)  # (batch_size, context_len, d_model)

    x = self.decoder(x=x, context=context, fx_intercept = fx_intercept)  # (batch_size, target_len, d_model)

    # Final linear layer output.
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits




class TransformerDecoder(tf.keras.Model):
  def __init__(self, *, d_model,  dff,
               num_layers_dec, num_heads_dec, skip_connection_dec,
               target_vocab_size, pos_encoding_length, dropout_rate=0.1,  **kwargs):
    super().__init__(**kwargs)


    self.decoder = Decoder(num_layers=num_layers_dec, d_model=d_model,
                           num_heads=num_heads_dec, dff=dff, skip_connection=skip_connection_dec,
                           vocab_size=target_vocab_size,
                           pos_encoding_length=pos_encoding_length,
                           dropout_rate=dropout_rate,
                           name="decoder")

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs,  fx_intercept = None):

    context, x  = inputs.get("input_text_encoder", None), inputs["input_text_decoder"]

    x = self.decoder(x = x, context = context, fx_intercept = fx_intercept)  # (batch_size, target_len, d_model)

    # Final linear layer output.
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits




# -------------------------------Optimizer && Loss-------------------------


# Use the Adam optimizer with a custom learning rate scheduler according to the formula in the original Transformer paper.
# lrate=d_model**−0.5∗min(step_num−0.5,step_num⋅warmup_steps−1.5)
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000, min_lr = 1e-4):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    self.min_lr = min_lr

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    lr   = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    return tf.math.maximum(lr, self.min_lr)




# Set up the loss and metrics
# Since the target sequences are padded, it is important to apply a padding mask when calculating the loss. Use the cross-entropy loss function (tf.keras.losses.SparseCategoricalCrossentropy):

def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss


def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)







def get_compiled_model(_train_conf: TransformerConfig, min_lr = 1e-4):
  transformer = None
  if _train_conf.encoderDecoder:
    transformer = Transformer(
      d_model=_train_conf.d_model,
      dff=_train_conf.dff,

      num_layers_enc=_train_conf.num_layers_enc,
      num_layers_dec=_train_conf.num_layers_dec,
      num_heads_enc = _train_conf.num_heads_enc,
      num_heads_dec = _train_conf.num_heads_dec,
      skip_connection_enc=_train_conf.skip_connection_enc,
      skip_connection_dec=_train_conf.skip_connection_dec,

      input_vocab_size=_train_conf.vocab_size,
      target_vocab_size=_train_conf.vocab_size,
      pos_encoding_length=_train_conf.pos_encoding_length,
      dropout_rate=_train_conf.dropout_rate,
      name=_train_conf.name)
  else:
    transformer = TransformerDecoder(
      d_model=_train_conf.d_model,
      dff=_train_conf.dff,

      num_layers_dec=_train_conf.num_layers_dec,
      num_heads_dec = _train_conf.num_heads_dec,
      skip_connection_dec=_train_conf.skip_connection_dec,

      target_vocab_size=_train_conf.vocab_size,
      pos_encoding_length=_train_conf.pos_encoding_length,
      dropout_rate=_train_conf.dropout_rate,
      name=_train_conf.name)

  #Instantiate the optimizer (in this example it's tf.keras.optimizers.Adam):
  learning_rate = CustomSchedule(_train_conf.d_model, min_lr = min_lr)
  optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,  epsilon=1e-9)
  perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)


  transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy, perplexity])

  return transformer, optimizer
