""" Bi-LSTM with max-pooling for learning sentence vectors.

The model is based on the paper:

  "Supervised Learning of Universal Sentence Representations "
  "from Natural Language Inference Data"
  Alexis Conneau, Douwe Kiela, Holger Schwenk, Loic Barrault, Antoine Bordes.
  https://arxiv.org/abs/1705.02364
"""

import tensorflow as tf

class InferModel(object):
    def __init__(self, config, mode="train"):
        """Basic setup. The actual TensorFlow graph is constructed in build().

        Args:
          config: Object containing configuration parameters.
          mode: "train" "eval" or "encode".

        Raises:
          ValueError: If mode is invalid.
        """
        if mode not in ["train", "eval", "encode"]:
            raise ValueError("Unrecognized mode: %s" % mode)

        self.config = config
        self.mode = mode

        # Initializer used for non-recurrent weights.
        self.uniform_initializer = tf.random_uniform_initializer(
            minval=-self.config.uniform_init_scale,
            maxval=self.config.uniform_init_scale)

        # Input sentences represented as sequences of word ids. "text" and "hypothesis"
        # are source sentences, labels is the relationship bwtween text and hypothesis.
        self.text_ids = None  # shape: [batch_size, padded_length]
        self.hypothesis_ids = None  # shape: [batch_size, padded_length]
        self.labels = None  # shape = [batch_size,]

        # Boolean masks distinguishing real words (1) from padded words (0).
        # Each is an int32 Tensor with shape [batch_size, padded_length].
        self.text_mask = None
        self.hypothesis_mask = None
        
        # Input sentences represented as sequences of word embeddings.
        self.text_emb = None  # shape: [batch_size, padded_length, emb_dim]
        self.hypothesis_emb = None  # shape: [batch_size, padded_length, emb_dim]

        # Used for load the pre-trained word embeddings from disk.
        self.word_emb_placeholder = tf.placeholder(
            tf.float32,
            [self.config.vocab_size, self.config.word_embedding_dim],
            name="word_embedding_placeholder")
        self.word_emb_assign_op = None

        # The output from the sentence encoder.
        self.text_vectors = None  # shape: [batch_size, num_hidden_units]
        self.hypothesis_vectors = None  # shape: [batch_size, num_hidden_units]

        # The cross entropy losses and accuracy. Used for evaluation.
        self.target_cross_entropy_loss = None
        self.eval_accuracy = None

        self.global_step = None

    def build_inputs(self, sent1_ids, sent1_mask, sent2_ids, sent2_mask, labels):
        """Builds the ops for reading input data.
        
        Args:
          sent1_ids: Input text represented as sequences of word ids.
          sent1_mask: Boolean masks of text, distinguish real words(1) from padded words(0).
          sent2_ids: Input hypothesis represented as sequences of word ids.
          sent2_mask: Boolean masks of hypothesis, distinguish real words(1) from padded words(0).
          labels: Each label can be 0(contradiction), 1(neutral), 2(entailment).

        Outputs:
          self.text_ids
          self.hypothesis_ids
          self.text_mask
          self.hypothesis_mask
          self.labels
        """
        if self.mode == "encode":
            self.text_ids = None
            self.hypothesis_ids = None
            self.text_mask = tf.placeholder(tf.int8, (None, None), name="text_mask")
            self.hypothesis_mask = None
            self.labels = None
        else:
            self.text_ids = sent1_ids
            self.hypothesis_ids = sent2_ids
            self.text_mask = sent1_mask
            self.hypothesis_mask = sent2_mask
            self.labels = labels

    def build_word_embeddings(self):
        """Builds the word embeddings.

        Inputs:
          self.text_ids
          self.hypothesis_ids

        Outputs:
          self.text_emb
          self.hypothesis_emb
          self.word_emb_assign_op
        """
        if self.mode == "encode":
            # Word embeddings are fed from an external vocabulary.
            self.text_emb = tf.placeholder(tf.float32, (
                None, None, self.config.word_embedding_dim), "encode_emb")
            self.hypothesis_emb = None
        else:
            word_emb = tf.get_variable(
                name="word_embedding",
                initializer=tf.constant(
                    0.0, tf.float32,
                    [self.config.vocab_size, self.config.word_embedding_dim]),
                trainable=False)
            self.word_emb_assign_op = word_emb.assign(self.word_emb_placeholder)
            self.text_emb = tf.nn.embedding_lookup(word_emb, self.text_ids)
            self.hypothesis_emb = tf.nn.embedding_lookup(word_emb, self.hypothesis_ids)

    def build_encoder(self):
        """Builds the sentence encoder.

        Inputs:
          self.text_emb
          self.hypothesis_emb
          self.text_mask
          self.hypothesis_mask

        Outputs:
          self.text_vectors
          self.hypothesis_vectors
        """
        if self.config.encoder_dim % 2 != 0:
            raise ValueError(
                "encoder_dim must be even when using a bidirectional encoder.")
        
        def _make_cell(num_units):
            cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units)
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, output_keep_prob=1 - self.config.encoder_dropout)
            return cell

        def _build_sentence_vectors(num_units, embedding, length):
            cell_fw = _make_cell(num_units)
            cell_bw = _make_cell(num_units)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=embedding,
                sequence_length=length,
                dtype=tf.float32)
            return outputs, states

        num_units = self.config.encoder_dim / 2

        with tf.variable_scope("encoder"):
            text_len = tf.to_int32(tf.reduce_sum(
                self.text_mask, axis=1, name="text_length"))
            text_outputs, _ = _build_sentence_vectors(
                num_units, self.text_emb, text_len)
            text_vectors = tf.concat(text_outputs, 2, name="text_vectors")
            self.text_vectors = tf.reduce_max(
                text_vectors, axis=1, name="text_max_pooling")

        with tf.variable_scope("encoder", reuse=True):
            hypothesis_len = tf.to_int32(tf.reduce_sum(
                self.hypothesis_mask, axis=1, name="hypothesis_length"))
            hypothesis_outputs, _ = _build_sentence_vectors(
                num_units, self.hypothesis_emb, hypothesis_len)
            hypothesis_vectors = tf.concat(
                hypothesis_outputs, 2, name="hypothesis_vectors")
            self.hypothesis_vectors = tf.reduce_max(
                hypothesis_vectors, axis=1, name="hypothesis_max_pooling")

    def build_loss_and_accuracy(self):
        """Build cross entropy loss.

        Inputs:
          self.text_vectors
          self.hypothesis_vectors

        Outputs:
          self.target_cross_entropy_loss
        """
        features = tf.concat([
            self.text_vectors,
            self.hypothesis_vectors,
            tf.abs(self.text_vectors - self.hypothesis_vectors),
            tf.multiply(self.text_vectors, self.hypothesis_vectors)], 1)

        def _linear_layer(inputs, output_dim):
            inputs_dim = inputs.get_shape().as_list()[-1]
            W = tf.get_variable(
                "W",
                shape=[inputs_dim, output_dim],
                dtype=tf.float32,
                initializer=self.uniform_initializer)
            b = tf.get_variable(
                "b",
                shape=[output_dim],
                dtype=tf.float32,
                initializer=self.uniform_initializer)
            outputs = tf.nn.xw_plus_b(inputs, W, b, name="out")
            return outputs

        def _linear_classifier(features, classifier_dim, num_classes):
            with tf.variable_scope("linear_layer_0"):
                features = _linear_layer(features, classifier_dim)

            with tf.variable_scope("linear_layer_1"):
                features = _linear_layer(features, classifier_dim)

            with tf.variable_scope("linear_layer_2"):
                logits = _linear_layer(features, num_classes)

            return logits

        def _nonlinear_classifier(features, classifier_dim, num_classes, dropout):
            with tf.variable_scope("nonlinear_layer_0"):
                features = _linear_layer(features, classifier_dim)
                features = tf.tanh(features)
                features = tf.nn.dropout(features, keep_prob=1 - dropout)

            with tf.variable_scope("nonlinear_layer_1"):
                features = _linear_layer(features, classifier_dim)
                features = tf.tanh(features)
                features = tf.nn.dropout(features, keep_prob=1 - dropout)

            with tf.variable_scope("nonlinear_layer_2"):
                logits = _linear_layer(features, num_classes)

            return logits

        if self.config.nonlinear_classifier:
            logits = _nonlinear_classifier(
                features,
                self.config.classifier_dim,
                self.config.num_classes,
                self.config.classifier_dropout)
        else:
            logits = _linear_classifier(
                features,
                self.config.classifier_dim,
                self.config.num_classes)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels, logits=logits)
        self.target_cross_entropy_loss = tf.reduce_mean(losses)
        # preds = tf.argmax(logits, axis=1)
        # self.eval_accuracy = tf.reduce_mean(tf.to_float(tf.equal(preds, self.labels)))
        correct = tf.nn.in_top_k(logits, self.labels, 1)
        self.eval_accuracy = tf.reduce_mean(tf.to_float(correct))

    def build_global_step(self):
        """Build the global step Tensor."""
        global_step = tf.get_variable(
            "global_step",
            dtype=tf.int32,
            initializer=tf.constant(0, dtype=tf.int32),
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        self.global_step = global_step

    def build(self, sent1_ids, sent1_mask, sent2_ids, sent2_mask, labels):
        """Creates all ops for training, evaluation or encoding."""
        self.build_inputs(sent1_ids, sent1_mask, sent2_ids, sent2_mask, labels)
        self.build_word_embeddings()
        self.build_encoder()
        self.build_loss_and_accuracy()
        self.build_global_step()