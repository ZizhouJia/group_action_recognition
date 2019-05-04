import torch
import os
import h5py
import readers
import numpy
import tensorflow as tf
import torch.utils.data
import os
from tensorflow import logging
from tensorflow import gfile
import itertools, datetime

class yt8mDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None,data_type = ''):
        super(yt8mDataset, self).__init__()
        data_dir = '/mnt/mmu/liuchang/hywData/yt8m/frame/' + data_type
        self.data_list = os.listdir(data_dir)
        self.range_list = []

        end_num = len(self.data_list) - 1
        while end_num >= 0:
            if not self.data_list[end_num].endswith('.tfrecord'):
                del self.data_list[end_num]
            else:
                self.data_list[end_num] = os.path.join(data_dir , self.data_list[end_num])
            end_num -= 1

        self.buff_input = None
        self.buff_label = None

        self.last_range_ind = -1
        self.data_type = data_type

    def get_range_ind(self,ind):
        print('range ind ' + str(ind))
        for i in range(len(self.range_list)):
            if ind >= self.range_list[i][0] and ind < self.range_list[i][1]:
                return i
        return -1

    def __getitem__(self, index):
        range_ind = self.get_range_ind(index)

        if range_ind == -1:
            start_ind = len(self.range_list)

            r_start_num = 0
            if start_ind > 0:
                r_start_num = self.range_list[-1][1]

            while range_ind == -1:
                self.buff_input,self.buff_label = gen_rec_data(self.data_list,rec_ind=start_ind)
                new_range = (r_start_num,self.buff_label.shape[0])
                self.range_list.append(new_range)
                range_ind = self.get_range_ind(index)
                # print('range ind ' + str(range_ind))

        elif range_ind != self.last_range_ind:
                self.buff_input, self.buff_label = gen_rec_data(self.data_list, rec_ind=range_ind)

        local_index = index - self.range_list[range_ind][0]
        self.last_range_ind = range_ind
        return self.buff_input[local_index],self.buff_label[local_index]

    def __len__(self):

        if self.data_type == 'train':
            return 5000000
        elif self.data_type == 'valid':
            return 1112356
        else:
            return 500000

def get_reader():

    feature_names = ['rgb','audio']
    feature_sizes = [1024,128]

    # Convert feature_names and feature_sizes to lists of values.
    # feature_names, feature_sizes = GetListOfFeatureNamesAndSizes(feature_names, feature_sizes)
    frame_features = True

    if frame_features:
        reader = readers.YT8MFrameFeatureReader(
            feature_names=feature_names, feature_sizes=feature_sizes)
    else:
        reader = readers.YT8MAggregatedFeatureReader(
            feature_names=feature_names, feature_sizes=feature_sizes)

    return reader

def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=1000,
                           num_epochs=None,
                           num_readers=1,
                           incl_val = False):
  """Creates the section of the graph which reads the training data.

  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_epochs: How many passes to make over the training data. Set to 'None'
                to run indefinitely.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  logging.info("Using batch size of " + str(batch_size) + " for training.")
  with tf.name_scope("train_input"):
    files = gfile.Glob(data_pattern)

    if not files:
      raise IOError("Unable to find training files. data_pattern='" +
                    data_pattern + "'.")
    if incl_val:
      print("Including Validation data!")
      files = files + gfile.Glob(data_pattern.replace("train", "validate"))
    logging.info("Number of training files: %s.", str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=num_epochs, shuffle=True)
    training_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]

    return tf.train.shuffle_batch_join(
        training_data,
        batch_size=batch_size,
        capacity=batch_size * 5,
        min_after_dequeue=batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)


def build_graph(reader,
                train_data_pattern,
                batch_size=1000,
                num_readers=1,
                num_towers = 1,
                num_epochs=None):
  """Creates the Tensorflow graph.

  This will only be called once in the life of
  a training model, because after the graph is created the model will be
  restored from a meta graph file rather than being recreated.

  Args:
    reader: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    train_data_pattern: glob path to the training data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
    base_learning_rate: What learning rate to initialize the optimizer with.
    optimizer_class: Which optimization algorithm to use.
    clip_gradient_norm: Magnitude of the gradient to clip to.
    regularization_penalty: How much weight to give the regularization loss
                            compared to the label loss.
    num_readers: How many threads to use for I/O operations.
    num_epochs: How many passes to make over the data. 'None' means an
                unlimited number of passes.
  """

  unused_video_id, model_input_raw, labels_batch, num_frames = (
      get_input_data_tensors(
          reader,
          train_data_pattern,
          batch_size=batch_size * num_towers,
          num_readers=num_readers,
          num_epochs=num_epochs))
  tf.summary.histogram("model/input_raw", model_input_raw)

  feature_dim = len(model_input_raw.get_shape()) - 1
  model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)
  tf.add_to_collection("input_batch", model_input)
  tf.add_to_collection("label_batch", tf.cast(labels_batch, tf.float32))



def gen_rec_data(data_list,rec_ind):
    batch_size = 10000
    num_towers = 1
    num_readers = 1
    num_epochs = 1
    reader = get_reader()
    train_data_pattern = data_list[rec_ind]

    with tf.device('/cpu:0'):
        with tf.Graph().as_default() as graph:
            # saver = build_model(reader,data_pattern=train_data_pattern)
            build_graph(reader=reader, train_data_pattern=train_data_pattern, batch_size=batch_size,
                        num_readers=num_readers, num_epochs=num_epochs, num_towers=num_towers)

            input_batch = tf.get_collection("input_batch")[0]
            label_batch = tf.get_collection("label_batch")[0]
            init_op = tf.global_variables_initializer()

            sv = tf.train.Supervisor(graph, init_op=init_op)

        # logging.info("%s: Starting managed session.", task_as_string(self.task))
        with sv.managed_session() as sess:
            [batch_val, label_val] = sess.run([input_batch, label_batch])
            batch_val = torch.from_numpy(batch_val)
            label_val = torch.from_numpy(label_val)

    return batch_val,label_val

if __name__ == "__main__":
    yt8m = yt8mDataset()


    source_trainloader = torch.utils.data.DataLoader(yt8m, batch_size=50, shuffle=False,
                                                     num_workers=1, drop_last=True)
    tmp_loader = iter(source_trainloader)
    for i in range(10):
        data,label = tmp_loader.next()
        print('data ' + str(data.shape))
        print('label ' + str(label.shape))
    c = 1
    #
    # for i, data_batch in enumerate(itertools.izip(source_trainloader)):
    #     c = 1