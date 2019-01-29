"""Train the sentence embedding model(BiLSTM with max pooling)."""

import tensorflow as tf

import os
import data_utils
import configuration
import infer_model

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_train_file_pattern", None,
                       "File pattern of sharded TFRecord files containing "
                       "tf.Example protos for training.")

tf.flags.DEFINE_string("input_valid_file_pattern", None,
                       "File pattern of sharded TFRecord files containing "
                       "tf.Example protos for validating.")

tf.flags.DEFINE_string("glove_file", None,
                       "The pre-trained glove embedding file.")

tf.flags.DEFINE_string("train_dir", None,
                       "Directory for saving and loading checkpoints.")

tf.logging.set_verbosity(tf.logging.INFO)

def main(_):
    if not FLAGS.input_train_file_pattern:
        raise ValueError("--input_train_file_pattern is required.")
    if not FLAGS.input_valid_file_pattern:
        raise ValueError("--input_valid_file_pattern is required.")
    if not FLAGS.glove_file:
        raise ValueError("--glove_file is required.")
    if not FLAGS.train_dir:
        raise ValueError("--train_dir is required.")

    if not tf.gfile.IsDirectory(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)

    model_config = configuration.ModelConfig()
    train_config = configuration.TrainingConfig()
    
    tf.logging.info("Load pre-trained Glove embeddings.")
    pretrained_emb = data_utils.load_pretrained_embeddings(
        FLAGS.glove_file, model_config.vocab_size)

    g = tf.Graph()
    with g.as_default():
         # Build training and valid dataset.
        training_dataset = data_utils.create_input_data(
            FLAGS.input_train_file_pattern,
            model_config.shuffle,
            model_config.batch_size)
        valid_dataset = data_utils.create_input_data(
            FLAGS.input_valid_file_pattern,
            model_config.shuffle,
            model_config.batch_size)
        iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                                   training_dataset.output_shapes)
        (next_text_ids, next_text_mask,
         next_hypothesis_ids, next_hypothesis_mask, next_label) = iterator.get_next()

        training_iterator_init = iterator.make_initializer(training_dataset)
        valid_iterator_init = iterator.make_initializer(valid_dataset)

        tf.logging.info("Building training graph.")

        learning_rate_placeholder = tf.placeholder(tf.float32, [], name="learning_rate")

        with tf.variable_scope("model"):
            model_config.encoder_dropout = 0.5
            model_config.classifier_dropout = 0.5
            model_train = infer_model.InferModel(model_config, mode="train")
            model_train.build(next_text_ids,
                              next_text_mask,
                              next_hypothesis_ids,
                              next_hypothesis_mask,
                              next_label)

        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate_placeholder)
        train_op = optimizer.minimize(
            model_train.target_cross_entropy_loss,
            global_step=model_train.global_step)

        with tf.variable_scope("model", reuse=True):
            model_config.encoder_dropout = 1.0
            model_config.classifier_dropout = 1.0
            model_valid = infer_model.InferModel(model_config, mode="eval")
            model_valid.build(next_text_ids,
                              next_text_mask,
                              next_hypothesis_ids,
                              next_hypothesis_mask,
                              next_label)

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

    with tf.Session(graph=g) as sess:
        # Initialize global variables.
        sess.run(init)
        # Assign pre-trained word embeddings to the model.
        sess.run(model_train.word_emb_assign_op,
                 feed_dict={model_train.word_emb_placeholder: pretrained_emb})

        lr = train_config.initial_learning_rate
        prev_accuracy = 0.0
        max_accuracy = 0.0
        epoch = 0
        while lr > train_config.learning_rate_threshold:
            # Initialize the iterator on training and valid dataset.
            sess.run(training_iterator_init)
            tf.logging.info("Epoch %d, learning rate: %f" % (epoch, lr))
            total_train_batch = 0
            total_train_loss = 0.0
            while True:
                try:
                    _, train_loss = sess.run(
                        [train_op, model_train.target_cross_entropy_loss],
                        feed_dict={learning_rate_placeholder: lr})
                    total_train_batch += 1
                    total_train_loss += train_loss
                    tf.logging.info("Batch %d, loss: %f" % (total_train_batch, train_loss))
                except tf.errors.OutOfRangeError:
                    break
            train_loss = total_train_loss / total_train_batch
            tf.logging.info("Train loss: %f" % train_loss)
            sess.run(valid_iterator_init)
            total_valid_batch = 0
            total_valid_loss = 0.0
            total_valid_accuracy = 0.0
            while True:
                try:
                    valid_loss, valid_accuracy = sess.run([
                        model_valid.target_cross_entropy_loss,
                        model_valid.eval_accuracy])
                    total_valid_batch += 1
                    total_valid_loss += valid_loss
                    total_valid_accuracy += valid_accuracy
                except tf.errors.OutOfRangeError:
                    break
            valid_loss = total_valid_loss / total_valid_batch
            valid_accuracy = total_valid_accuracy / total_valid_batch
            tf.logging.info("Validate loss: %f, accuracy: %f" % (valid_loss, valid_accuracy))
            if valid_accuracy > prev_accuracy:
                lr *= train_config.learning_rate_decay_factor
            else:
                lr /= 5
            if valid_accuracy > max_accuracy:
                max_accuracy = valid_accuracy
                saver.save(sess, FLAGS.train_dir, global_step=model_train.global_step)
            prev_accuracy = valid_accuracy

if __name__ == "__main__":
    tf.app.run()
