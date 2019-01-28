"""Convert text corpus to TFRecord format with Example protos.
Some methods in this module are adapted from tensorflow models/research/skip_thoughts.
"""

import json
import collections
import os

import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_files", None,
                       "Comma-separated list of input files, including "
                       "train, valid, and test dataset.")

tf.flags.DEFINE_string("glove_file", None,
                       "A pre-trained Glove embeddings file.")

tf.flags.DEFINE_string("output_dir", None, "Output directory.")

tf.flags.DEFINE_integer("num_words", 50000,
                        "Number of words to include in the output.")

tf.flags.DEFINE_integer("max_sentence_length", 30,
                        "If > 0, exclude sentences that exceeds this length.")

tf.flags.DEFINE_integer("train_output_shards", 100,
                        "Number of output shards for the training set.")

tf.flags.DEFINE_integer("validation_output_shards", 1,
                        "Number of output shards for the validation set.")

tf.flags.DEFINE_integer("test_output_shards", 1,
                        "Number of output shards for the test set.")

tf.logging.set_verbosity(tf.logging.INFO)

EOS = "<EOS>"
EOS_ID = 0
UNK = "<UNK>"
UNK_ID = 1

# def _build_vocab(input_file):
#     """Build the vocabulary based on a list of files.

#     Args:
#       input_file: An SNLI-format json file.

#     Returns:
#       A dictionary of word to id.
#     """
#     word_cnt = collections.Counter()
#     with tf.gfile.GFile(input_file, mode='r') as f:
#         for line in f:
#             json_line = json.loads(line)
#             sent1 = json_line.get("sentence1", "").strip(".")
#             sent2 = json_line.get("sentence2", "").strip(".")
#             word_cnt.update(sent1.split())
#             word_cnt.update(sent2.split())
#     sorted_items = word_cnt.most_common()
#     vocab = collections.OrderedDict()
#     vocab[EOS] = EOS_ID
#     vocab[UNK] = UNK_ID
#     for widx, item in enumerate(sorted_items):
#         vocab[item[0]] = widx + 2
#     tf.logging.info("Create vocab with %d words.", len(vocab))

#     vocab_file = os.path.join(FLAGS.output_dir, "vocab.txt")
#     with tf.gfile.GFile(vocab_file, mode="w") as f:
#         f.write("\n".join(vocab.keys()))
#     tf.logging.info("Wrote vocab file to %s", vocab_file)

#     word_cnt_file = os.path.join(FLAGS.output_dir, "word_count.txt")
#     with tf.gfile.GFile(word_cnt_file, mode="w") as f:
#         for w, c in sorted_items:
#           f.write("%s %d\n" % (w, c))
#     tf.logging.info("Wrote vocab file to %s", word_cnt_file)
#     return vocab

def _build_vocab(input_file):
    """Build the vocabulary based on a pre-trained Glove embeddings file.

    Args:
      input_file: A pre-trained Glove embeddings file.

    Returns:
      A dictionary of word to id.
    """
    vocab = collections.OrderedDict()
    vocab[EOS] = EOS_ID
    vocab[UNK] = UNK_ID
    i = 2
    with tf.gfile.GFile(input_file, "r") as f:
        for line in f:
            if i >= FLAGS.num_words:
                break
            toks = line.split()
            vocab[toks[0]] = i
            i += 1
    
    vocab_file = os.path.join(FLAGS.output_dir, "vocab.txt")
    with tf.gfile.GFile(vocab_file, mode="w") as f:
        f.write("\n".join(vocab.keys()))
    tf.logging.info("Wrote vocab file to %s", vocab_file)
    return vocab

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=[int(v) for v in value]))

def _sentence_to_ids(sentence, vocab):
    """Helper for converting a sentence (list of words) to a list of ids."""
    ids = [vocab.get(w, UNK_ID) for w in sentence]
    ids.append(EOS_ID)
    return ids

def _label_to_id(label):
    return {"contradiction": 0, "neutral": 1, "entailment": 2}.get(label)

def _create_serialized_example(sent1, sent2, label, vocab):
    """Helper for creating a serialized Example proto."""
    example = tf.train.Example(features=tf.train.Features(feature={
        "sentence1": _int64_feature(_sentence_to_ids(sent1, vocab)),
        "sentence2": _int64_feature(_sentence_to_ids(sent2, vocab)),
        "label": _int64_feature([_label_to_id(label)])
    }))
    return example.SerializeToString()

def _build_dataset(filename, vocab):
    """Build a dataset from an SNLI json file.

    Args:
      filename: An SNLI-format json file.
      vocab: A dictionary of word to id.

    Returns:
      A list of serialized Example protos.
    """
    serialized = []
    with tf.gfile.GFile(filename, mode="r") as f:
        for line in f:
            json_line = json.loads(line)
            sent1 = json_line.get("sentence1", "").strip(".").split()
            sent2 = json_line.get("sentence2", "").strip(".").split()
            if FLAGS.max_sentence_length and (
                len(sent1) >= FLAGS.max_sentence_length
                or len(sent2) >= FLAGS.max_sentence_length):
                continue
            label = json_line.get("gold_label", "")
            # If there is no gold label, SNLI uses "-" insted.
            if label not in ["contradiction", "neutral", "entailment"]:
                continue
            serialized.append(_create_serialized_example(sent1, sent2, label, vocab))
    return serialized

def _write_shard(filename, dataset, indices):
    """Writes a TFRecord shard."""
    with tf.python_io.TFRecordWriter(filename) as writer:
        for j in indices:
            writer.write(dataset[j])

def _write_dataset(name, dataset, num_shards):
    """Writes a sharded TFRecord dataset.

    Args:
      name: Name of the dataset (e.g. "train").
      dataset: List of serialized Example protos.
      num_shards: The number of output shards.
    """
    shuffled_indices = np.random.permutation(len(dataset))
    borders = np.int32(np.linspace(0, len(shuffled_indices), num_shards + 1))
    for i in range(num_shards):
        filename = os.path.join(
            FLAGS.output_dir, "%s-%.5d-of-%.5d" % (name, i, num_shards))
        indices = shuffled_indices[borders[i]:borders[i + 1]]
        _write_shard(filename, dataset, indices)
        tf.logging.info("Wrote dataset indices [%d, %d) to output shard %s",
                        borders[i], borders[i + 1], filename)

def main(_):
    if not FLAGS.input_files:
        raise ValueError("--input_files is required.")
    if not FLAGS.glove_file:
        raise ValueError("--glove_file is requires.")
    if not FLAGS.output_dir:
        raise ValueError("--output_dir is required.")

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    
    input_files = FLAGS.input_files.split(",")
    if len(input_files) != 3:
        raise ValueError("Train, validate and test datasets are all needed.")

    vocab = _build_vocab(FLAGS.glove_file)  # Use pre-trained Glove to build vocab.

    train_dataset = _build_dataset(input_files[0], vocab)
    _write_dataset("train", train_dataset, FLAGS.train_output_shards)

    valid_dataset = _build_dataset(input_files[1], vocab)
    _write_dataset("valid", valid_dataset, FLAGS.validation_output_shards)

    test_dataset = _build_dataset(input_files[2], vocab)
    _write_dataset("test", test_dataset, FLAGS.test_output_shards)

if __name__ == "__main__":
    tf.app.run()
