# read files in Youtube-8m in .tfrecord

import json
import os
import time


import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.lib.io import file_io
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.client import device_lib
import itertools


def get_features_from_file(file_path):
    '''
    :param file_path: path for a file of .tfrecord
    :return: id, labels, features for each frame in this file
    '''
    features_rgb = []
    features_audio = []
    labels = []
    ids = []
    for example in tf.python_io.tf_record_iterator(file_path):

        parsed_example = tf.train.Example.FromString(example)

        ids.append(parsed_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8'))
        labels.append(parsed_example.features.feature['labels'].int64_list.value)

        tf_seq_example = tf.train.SequenceExample.FromString(example)
        n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)
        sess = tf.InteractiveSession()
        rgb_frame = []
        audio_frame = []

        for i in range(n_frames):
            rgb_frame.append(tf.cast(tf.decode_raw(
                tf_seq_example.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0], tf.uint8)
                , tf.float32).eval())
            audio_frame.append(tf.cast(tf.decode_raw(
                tf_seq_example.feature_lists.feature_list['audio'].feature[i].bytes_list.value[0], tf.uint8)
                , tf.float32).eval())

        sess.close()
        features_rgb.append(rgb_frame)
        features_audio.append(audio_frame)



    return ids, labels, features_rgb, features_audio



FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
  #flags.DEFINE_string("files_dir", "../AIdata/", "The directory of the .tfrecords.")

  record_name = './validate2L.tfrecord'

  get_features_from_file(record_name)



'''
data：
--------------
context: {
  feature: {
    key  : "id"
    value: {
      bytes_list: {
        value: [Video id. Can be translated to YouTube ID (link).]
      }
    }
  }
  feature: {
    key  : "labels"
      value: {
        int64_list: {
          value: [1, 522, 11, 172] # The meaning of the labels can be found here.
        }
      }
    }
}

feature_lists: {
  feature_list: {
    key  : "rgb"
    value: {
      feature: {
        bytes_list: {
          value: [1024 8bit quantized features]
        }
      }
      feature: {
        bytes_list: {
          value: [1024 8bit quantized features]
        }
      }
      ... # Repeated for every second of the video, up to 300
  }
  feature_list: {
    key  : "audio"
    value: {
      feature: {
        bytes_list: {
          value: [128 8bit quantized features]
        }
      }
      feature: {
        bytes_list: {
          value: [128 8bit quantized features]
        }
      }
    }
    ... # Repeated for every second of the video, up to 300
  }

}
'''