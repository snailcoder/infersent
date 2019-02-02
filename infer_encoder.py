""" Use InferModel to encode sentences."""

import collections

import tensorflow as tf

import infer_model
import data_utils

class InferEncoder(object):
    def __init__(self):
        self.word_embeddings = None

    def _load_word_embeddings(self, vocab_file, embedding_file):
        with tf.gfile.GFile(vocab_file, mode="r") as f:
            lines = f.readlines()
        vocab = [line.strip() for line in lines]
        tf.logging.info("Loaded vocabulary with %d words.", len(vocab))

        embeddings = data_utils.load_pretrained_embeddings(
            embedding_file, len(vocab))

        self.word_embeddings = collections.OrderedDict(zip(vocab, embeddings))

    def load_model(self, model_config, vocabulary_file, embedding_matrix_file,
                   checkpoint_path):
        """Loads a infersent model.

        Args:
          model_config: Object containing parameters for building the model.
          vocabulary_file: Path to vocabulary file containing a list of newline-
            separated words where the word id is the corresponding 0-based index in
            the file.
          embedding_matrix_file: Path to a serialized numpy array of shape
            [vocab_size, embedding_dim].
          checkpoint_path: InferModel checkpoint file or a directory
            containing a checkpoint file.
        """
        self._load_word_embeddings(vocabulary_file, embedding_matrix_file)

    def buil_encoder(self):
        pass