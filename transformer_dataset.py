
import time
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib
import pickle
import logging

import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_text
import tensorflow_datasets as tfds
import keras_nlp

import datasets
from datasets.utils.file_utils import get_datasets_user_agent




USER_AGENT = get_datasets_user_agent()



class CustomDatasetGen:
  """
    # https://www.tensorflow.org/datasets/catalog/overview
    # https://huggingface.co/docs/datasets/index
  """

  FEATURE_CONCATENATION_TOKEN: str  = "\n  #######   \n"

  class DSORIGIN(Enum):
    UNKN = 0
    TFDS = 1
    HuggingFace = 2



  def __init__(self, name: str = "", path:str = "", ds_sub_name = None, origin:DSORIGIN = DSORIGIN.TFDS, supervised:bool=False, hasTest: bool = False, hasValidation: bool = False,
               tokenizer = None, trainconfig = None,  fromGen: bool = True, skip_first = None, hasImage=None):


    self.name = name;

    self.path = path; #path of DS
    self.ds_sub_name = ds_sub_name # name of subfolder of DS
    self.origin = origin;
    self.supervised = supervised

    self.hasTest = hasTest
    self.hasValidation = hasValidation
    self.metadata  = None



    self.fromGen = fromGen # wheter or not  it is an iterable DS
    self.genItems = None   # list of key retrieve in tfds_gen
    self.gen_output_signature = None # tf output_signature of genItems

    self.tokenizer   =  tokenizer
    self.batch_size  =  trainconfig.batch_size   if trainconfig != None else 32
    self.buffer_size =  trainconfig.buffer_size  if trainconfig != None else 128
    self.max_tokens  =  trainconfig.max_tokens   if trainconfig != None else 128
    self.encoderDecoder = trainconfig.encoderDecoder if trainconfig != None  else False #if False --> Decoder only

    #img hyper param
    self.hasInpImage = trainconfig.hasInpImage if trainconfig != None  else False
    self.hasOutImage = trainconfig.hasOutImage if trainconfig != None  else False
    self.hasImage = self.hasInpImage or self.hasOutImage
    if hasImage != None: self.hasImage = hasImage #overwrite param with ds arg

    self.img_width  = trainconfig.img_width  if trainconfig != None  else 256
    self.img_height = trainconfig.img_height if trainconfig != None  else 256


    self.skip_first = skip_first
    self.fx_unpack_data = lambda x : x






  def download_data(self, ):
    if self.origin == self.DSORIGIN.TFDS: self.download_from_tfds();
    if self.origin == self.DSORIGIN.HuggingFace: self.download_from_huggingface();
    return None

  def download_from_tfds(self, ):
    self.all_data, self.metadata = tfds.load(self.path, with_info=True,  as_supervised=self.supervised, try_gcs=True)
    return None


  def download_from_huggingface(self, ):
    self.fromGen = True
    _iterable_ds = datasets.load_dataset(path = self.path, name = self.ds_sub_name, split="train", streaming=True)
    # self.metadata = { "features" : _iterable_ds["features"],   "n_shards" : _iterable_ds["n_shards"] }

    self.all_data = { "train" :  tf.data.Dataset.from_generator( self.tfds_gen( _iterable_ds  ), output_signature= self.gen_output_signature )  };
    if self.hasTest:       self.all_data['test']       = tf.data.Dataset.from_generator( self.tfds_gen( datasets.load_dataset(path = self.path, name = self.ds_sub_name, split="test",       streaming=True)), output_signature= self.gen_output_signature );
    if self.hasValidation: self.all_data['validation'] = tf.data.Dataset.from_generator( self.tfds_gen( datasets.load_dataset(path = self.path, name = self.ds_sub_name, split="validation", streaming=True)), output_signature= self.gen_output_signature );

    return None


  def tfds_gen(self, _iterable_ds):

    def _gen():
      for elt in _iterable_ds:
        elt =  self.fx_unpack_data( elt )
        _skip_row = elt.get("skip_row", False)
        if _skip_row: continue
        else: yield {  k : elt[k] for k in self.genItems } if self.genItems != None else elt

    return _gen




  def tokenize_inp(self, args:dict):

    _decoder_inp = args.get("_decoder_inp");
    _labels      = args.get("_labels", None);
    _encoder_inp = args.get("_encoder_inp", None);

    (_inp, _outp) = (_encoder_inp, _decoder_inp) if _encoder_inp != None else (tf.identity(_decoder_inp), _decoder_inp)


    _inp = self.tokenizer(_inp, sequence_length = self.max_tokens)    # Output is ragged.
    _inp = _inp[:, :self.max_tokens]    # Trim to MAX_TOKENS.


    _outp = self.tokenizer(_outp, sequence_length = self.max_tokens + 1);
    _outp = _outp[:, :(self.max_tokens+1)]
    _deco_inp = _outp[:, :-1]  # Drop the [END] tokens


    if _labels == None:
      _labels   = _outp[:, 1:] # Drop the [START] tokens;

    else:
      _labels  = self.tokenizer(_labels, sequence_length = self.max_tokens + 1)
      _labels  = _labels[:, :(self.max_tokens+1)]
      _labels  = _labels[:, 1:]   # Drop the [START] tokens

    _inputs = {"input_text_decoder": _deco_inp}
    if self.encoderDecoder:
      _inputs["input_text_encoder"] = _inp

    return _inputs, _labels



  def img_preprocessing(self, _img):
    #tdod: add image preprocessing and augmention
    # _img = tf.convert_to_tensor(_img, dtype=tf.uint8)

    return _img;


  def text_preprocessing(self, x):
    #todo: add text preprocessing and augmention
    if self.tokenizer == None:  return {"input_text_decoder": x}, x
    x = self.tokenize_inp(x)

    return x;


  def preprocess(self, inputs ):

    _inputs, _text_labels   = self.text_preprocessing( inputs )


    if self.hasImage:
      _image = self.img_preprocessing(  inputs["image"]  )
      if self.hasInpImage:
        _inputs["input_image"] = _image


    return _inputs, _text_labels


  def make_batches(self, ds):

    return (
        ds
        .shuffle(self.buffer_size)
        .batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        .map(self.preprocess, tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        )



  def build_tfds(self, ):

    self.download_data( );

    self.train_data =  self.make_batches(self.all_data['train'])
    if self.hasTest:       self.test_data =  self.make_batches(self.all_data['test'])
    if self.hasValidation: self.val_data  =  self.make_batches(self.all_data['validation'])

    return None


  def print_exemple(self, show_token=False, num_exemples = 2):
    for _inp_ex, _labels_ex in self.train_data.take(1):
      self._inp_ex = _inp_ex; self._labels_ex = _labels_ex

      _enc_ex_str = _inp_ex.get("input_text_encoder", None);
      _dec_ex_str = _inp_ex.get("input_text_decoder", None);
      if _enc_ex_str != None: _enc_ex_str =  _enc_ex_str[:num_exemples]
      if _dec_ex_str != None: _dec_ex_str =  _dec_ex_str[:num_exemples]

      _text_labels_ex_str = None
      if _labels_ex != None: _text_labels_ex_str =  _labels_ex[:num_exemples]

      none_list = [None for _ in range(num_exemples) ]
      _enc_ex_token , _dec_ex_token, _text_labels_ex_token = none_list[:], none_list[:], none_list[:]


      _image = None;
      if self.hasImage and self.hasInpImage: _image = self._inp_ex.get( "input_image", None   )


      if self.tokenizer != None:
        _enc_ex_token , _dec_ex_token, _text_labels_ex_token = _enc_ex_str, _dec_ex_str, _text_labels_ex_str
        if _enc_ex_str != None: _enc_ex_str = self.tokenizer.detokenize( _enc_ex_str )
        if _dec_ex_str != None: _dec_ex_str = self.tokenizer.detokenize( _dec_ex_str )
        if _text_labels_ex_str != None: _text_labels_ex_str = self.tokenizer.detokenize( _text_labels_ex_str )


      if _enc_ex_str != None or _dec_ex_str != None:
        print('--------------------------------------> Text Inputs examples:')
        if _enc_ex_str != None:
          print('----------> Encoder Text Inputs examples:')
          for _idx, (_token, ex_enc_inp) in enumerate(zip( _enc_ex_token, _enc_ex_str.numpy())):
            print("++++> ex: ", _idx)
            if show_token: print(_token)
            print(ex_enc_inp.decode('utf-8'))
          print()

        if _dec_ex_str != None:
          print('----------> Decoder Text Labels examples:')
          for _idx, (_token, ex_dec_inp) in enumerate(zip( _dec_ex_token, _dec_ex_str.numpy())):
            print("++++> ex: ", _idx)
            if show_token: print(_token)
            print(ex_dec_inp.decode('utf-8'))
          print()



      if _text_labels_ex_str != None:
        print('--------------------------------------> Text Labels examples:')
        for _idx, (_token, _text_labels) in enumerate(zip( _text_labels_ex_token, _text_labels_ex_str.numpy())):
          print("++++> ex: ", _idx)
          if show_token: print(_token)
          print(_text_labels.decode('utf-8'))
        print()


      if self.hasImage and self.hasInpImage != None:
        print('--------------------------------------> Image Input examples:')
        fig = plt.figure(figsize=(5, 4 * num_exemples))
        fig.tight_layout()
        for _img_idx in range(num_exemples):
          plt.subplot(1, num_exemples, _img_idx + 1 )
          plt.axis('off')
          plt.imshow(_image[_img_idx])
        print()





  @classmethod
  def fetch_single_image(self, image_url, rsz = None, timeout=0.65, retries=0):
    image = None
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
                if rsz != None:
                  image = tf.image.resize(images = image, size = rsz , method=tf.image.ResizeMethod.BICUBIC, preserve_aspect_ratio=False, antialias=True, name=None)
            break
        except Exception as exc:
            image = None
    return image









# ------------------ TFDS DS --------------------------------


def get_lambada_text_understanding( trainconfig = None,  tokenizer=None ):
  def fx(*args): return args[0]["passage"], args[0]["passage"]

  _tf_ds = CustomDatasetGen(name = "lambada_text_understanding", path="lambada", origin = "tfds", supervised=False, trainconfig = trainconfig,  hasTest = True, tokenizer=tokenizer )
  _tf_ds.fx_unpack_data = fx
  _tf_ds.build_tfds()
  return _tf_ds




def get_billsum_text_summuray( trainconfig = None,  tokenizer=None ):
  def fx(*args): return args[0]["title"] +  args[0]["text"]  , args[0]["summary"]

  _tf_ds = CustomDatasetGen(name = "billsum_text_summuray", path="scientific_papers/pubmed", origin = "tfds", supervised=True, trainconfig = trainconfig,  hasTest = True, tokenizer=tokenizer )
  _tf_ds.fx_unpack_data = fx
  _tf_ds.build_tfds()
  return _tf_ds











# ------------------ HUGGINGFACE DS --------------------------------


def get_orca_instruction( trainconfig = None,  tokenizer=None  ):

  def fx(*args): return { "_decoder_inp": args[0]["system_prompt"] +  "\n" + args[0]["question"],  "_labels" : args[0]["response"], "skip_row": False }


  out_spec = { "_decoder_inp" : tf.TensorSpec(shape=(), dtype=tf.string, name="_decoder_inp"), "_labels" : tf.TensorSpec(shape=(), dtype=tf.string, name="_labels"),  "skip_row": tf.TensorSpec(shape=(), dtype=tf.bool, name="skip_row" ) }

  _tf_ds = CustomDatasetGen(name = "orca_flan_instruction", path="Open-Orca/OpenOrca", origin = CustomDatasetGen.DSORIGIN.HuggingFace, supervised=True, trainconfig = trainconfig,  hasTest = False, tokenizer=tokenizer, hasImage=False )
  _tf_ds.fx_unpack_data = fx
  _tf_ds.gen_output_signature = out_spec
  # _tf_ds.genItems = [ "system_prompt", "question",  "response" ]

  _tf_ds.build_tfds()

  # _tf_ds.print_exemple()

  return _tf_ds







def wikipedia( trainconfig = None,  tokenizer=None  ):
  out_spec = { "_decoder_inp":  tf.TensorSpec(shape=(), dtype=tf.string, name="_decoder_inp" ),  "skip_row": tf.TensorSpec(shape=(), dtype=tf.bool, name="skip_row" )  }


  
  if trainconfig != None and trainconfig.encoderDecoder:
    def fx(*args): return { "_decoder_inp": args[0]["title"], "_encoder_inp": args[0]["text"],  "skip_row": False  }
    out_spec[ "_encoder_inp"  ] = tf.TensorSpec(shape=(), dtype=tf.string, name="_encoder_inp")
  else:
    def fx(*args): return { "_decoder_inp": args[0]["title"] +  "\n" + args[0]["text"],         "skip_row": False  }



  _tf_ds = CustomDatasetGen(name = "wikimedia", path="wikimedia/wikipedia", ds_sub_name= "20231101.en", origin = CustomDatasetGen.DSORIGIN.HuggingFace, supervised=True, trainconfig = trainconfig,  hasTest = False, tokenizer=tokenizer, hasImage=False )
  _tf_ds.fx_unpack_data = fx
  _tf_ds.gen_output_signature = out_spec

  _tf_ds.build_tfds()

  # _tf_ds.print_exemple()

  return _tf_ds








# ------------------ Multimodal Text-Image HUGGINGFACE DS --------------------------------


# def get_coco_text2img_tfds( trainconfig = None,  tokenizer=None ):
#   def fx(*args): return args[0]["captions"], args[0]["image"]    #always return (text, image)

#   out_spec = { "captions":  tf.TensorSpec(shape=None, dtype=tf.string, name="captions" ), "image" :   tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8, name="image" ),  }

#   _tf_ds = TextImage(name = "coco", path="coco_captions", origin = TextImage.DSORIGIN.TFDS, supervised=True, trainconfig = trainconfig,  hasTest = True, tokenizer=tokenizer )
#   _tf_ds.fx_unpack_data = fx
#   _tf_ds.gen_output_signature = out_spec
#   _tf_ds.genItems = [ "captions", "image",  ]



#   _tf_ds.build_tfds()
#   return _tf_ds




def get_coco_text2img_hugging( trainconfig = None,  tokenizer=None ):
  def fx(*args): #always return (text, image)
    return ( args[0]["sentences"]["raw"],  args[0]["image"] ), False

  out_spec = ( tf.TensorSpec(shape=(), dtype=tf.string, name="captions" ),   tf.TensorSpec(shape=(None, None, None), dtype=tf.uint8, name="image" ) )

  _tf_ds = CustomDatasetGen(name = "coco", path="HuggingFaceM4/COCO", origin = CustomDatasetGen.DSORIGIN.HuggingFace, supervised=True, trainconfig = trainconfig,  hasTest = True, tokenizer=tokenizer,  )
  _tf_ds.fx_unpack_data = fx
  _tf_ds.gen_output_signature = out_spec
  # _tf_ds.genItems = [ "sentences", "image",  ]


  _tf_ds.build_tfds()
  return _tf_ds







def get_laion_coco_aesthetic( trainconfig = None,  tokenizer=None ):
  # https://huggingface.co/datasets/guangyil/laion-coco-aesthetic?row=23
  def fx(*args): #always return (text, image)
    _img = args[0]["url"]

    _img = CustomDatasetGen.fetch_single_image(args[0]["url"],  )

    print()

    return  { "_decoder_inp": args[0]["caption"],   "image" :  _img,  "skip_row": _img == None }

  out_spec = { "_decoder_inp" : tf.TensorSpec(shape=(), dtype=tf.string, name="_decoder_inp" ), "image" :   tf.TensorSpec(shape=(None, None, None), dtype=tf.uint8, name="image" ) }

  _tf_ds = CustomDatasetGen(name = "laion-coco-aesthetic", path="guangyil/laion-coco-aesthetic", origin = CustomDatasetGen.DSORIGIN.HuggingFace, supervised=True, hasTest = False, trainconfig = trainconfig,   tokenizer=tokenizer,  )
  _tf_ds.fx_unpack_data = fx
  _tf_ds.gen_output_signature = out_spec


  _tf_ds.build_tfds()
  return _tf_ds



def get_conceptual_captions( trainconfig = None,  tokenizer=None ):
  # https://huggingface.co/datasets/conceptual_captions?row=34


  _tf_ds = CustomDatasetGen(name = "conceptual_captions", path="conceptual_captions",ds_sub_name="unlabeled", origin = CustomDatasetGen.DSORIGIN.HuggingFace, supervised=True, hasValidation = True, trainconfig = trainconfig,   tokenizer=tokenizer,  )

  out_spec = { "_decoder_inp" : tf.TensorSpec(shape=(), dtype=tf.string, name="_decoder_inp"), "image" : tf.TensorSpec(shape=(None, None, None), dtype=tf.uint8, name="image"), "skip_row": tf.TensorSpec(shape=(), dtype=tf.bool, name="skip_row" ) }

  def fx(*args):
    _img = args[0]["image_url"]
    _img = _tf_ds.fetch_single_image(image_url = _img, rsz=(_tf_ds.img_height, _tf_ds.img_width))

    return  { "_decoder_inp": args[0]["caption"],  "image" :  _img,  "skip_row": _img == None }


  _tf_ds.fx_unpack_data = fx
  _tf_ds.gen_output_signature = out_spec
  # _tf_ds.genItems = [ "sentences", "image",  ]



  _tf_ds.build_tfds()
  return _tf_ds




