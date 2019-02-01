# infersent

This is a Tensorflow version of [InferSent](https://github.com/facebookresearch/InferSent).
For simplicity, this project only implement the BiLSTM with max pooling model, based on
[Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364).

## How to train the model?
1. Download [SNLI](https://nlp.stanford.edu/projects/snli/snli_1.0.zip) dataset and
[GloVe vectors trained on Common Crawl 840B with 300 dimensions](http://nlp.stanford.edu/data/glove.840B.300d.zip).

2. Pre-process SNLI data, create dataset for training and validating, and store them in TFRecord files:
```bash
python preprocess_dataset.py --glove_file /path/to/your/glove --input_files /path/to/your/snli/snli_1.0_train.jsonl,/path/to/your/snli/snli_1.0_dev.jsonl,/path/to/your/snli/snli_1.0_test.jsonl --output_dir /path/to/save/tfrecords
```

3. Train the InferModel:
```bash
python train.py --glove_file /path/to/your/glove --input_train_file_pattern "/path/to/save/tfrecords/train-?????-of-?????" --input_valid_file_pattern "/path/to/save/tfrecords/valid-?????-of-?????" --train_dir /path/to/save/checkpoints
```

## Experiment result
With the default settings in configurations.py, I obtained a dev accuracy of 83.72% in epoch 5,
83.37% in epoch 10.

## Tips for fine-tuning
Don't dropout too much. When the classifier and encoder dropout are both set to 0.5, both the
train and the dev accuracy is decreased below 70%.