""" Data utils.
Some methods in this module are adapted from tensorflow models/research/skip_thoughts.
"""
import collections

import tensorflow as tf
import numpy as np

def create_input_data(file_pattern, shuffle, batch_size):
    """Fetches string values from disk into tf.data.Dataset.

    Args:
      file_pattern: Comma-separated list of file patterns (e.g.
          "/tmp/train_data-?????-of-00100", where '?' acts as a wildcard that
          matches any character).
      shuffle: Boolean; whether to randomly shuffle the input data.
      batch_size: Batch size.

    Returns:
      A dataset read from TFRecord files.
    """
    data_files = []
    for pattern in file_pattern.split(","):
        data_files.extend(tf.gfile.Glob(pattern))
    if not data_files:
        tf.logging.fatal("Found no input files matching %s", file_pattern)
    else:
        tf.logging.info("Prefetching values from %d files matching %s",
                        len(data_files), file_pattern)

    dataset = tf.data.TFRecordDataset(data_files)

    def _parse_record(record):
        features = {
            "sentence1": tf.VarLenFeature(dtype=tf.int64),
            "sentence2": tf.VarLenFeature(dtype=tf.int64),
            "label": tf.FixedLenFeature((), tf.int64, 1)
        }
        parsed_features = tf.parse_single_example(record, features)

        def _sparse_to_dense(sparse):
            ids = tf.sparse.to_dense(sparse, default_value=0)  # Pad with zeros.
            mask = tf.sparse.to_dense(
                tf.sparse.SparseTensor(
                    sparse.indices, tf.ones_like(sparse.values), sparse.dense_shape))
            return ids, mask

        sent1_ids, sent1_mask = _sparse_to_dense(parsed_features["sentence1"])
        sent2_ids, sent2_mask = _sparse_to_dense(parsed_features["sentence2"])

        return sent1_ids, sent1_mask, sent2_ids, sent2_mask, parsed_features["label"]

    dataset = dataset.map(_parse_record)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=([None], [None], [None], [None], []))
    # If you want to iterate all epochs at once without information about
    # end of individual epochs, you can use dataset.repeat().
    # However, if you want to be informed about ending each of epoch,
    # dataset.repeat() should not be called.
    # dataset = dataset.repeat()  # Repeat the input indefinitely.
    return dataset

def load_pretrained_embeddings(input_file, num_words):
    """Load embeddings from a pre-trained Glove file.

    Args:
      input_file: A pre-trained word vectors file.

    Returns:
      A numpy array of pre-trained word embeddings.
    """
    emb_array = []
    i = 2  # 0 and 1 are reserved for EOS and UNK.
    with tf.gfile.GFile(input_file, "r") as f:
        for line in f:
            if i >= num_words:
                break
            toks = line.split()
            emb = np.array([float(v) for v in toks[1:]])
            emb_array.append(emb)
            i += 1
    EOS_emb = [0.0] * len(emb_array[0])
    UNK_emb = [0.0] * len(emb_array[0])

    embeddings = np.array([EOS_emb, UNK_emb] + emb_array)
    return embeddings