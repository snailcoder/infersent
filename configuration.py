"""Default configuration for model architecture and training."""

class ModelConfig(object):
    """Wrapper class for model hyperparameters."""
    def __init__(self):
        self.vocab_size = 50000
        self.word_embedding_dim = 300
        # Number of lstm hidden units.
        self.encoder_dim = 2048
        # Number of classifier hidden units.
        self.classifier_dim = 512
        self.batch_size = 64
        # Whether to randomly shuffle the input data.
        self.shuffle = True
        self.uniform_init_scale = 0.1
        self.num_classes = 3
        self.nonlinear_classifier = True
        self.encoder_dropout = 0.0
        self.classifier_dropout = 0.0

class TrainingConfig(object):
    """Wrapper class for training hyperparameters."""
    def __init__(self):
        self.initial_learning_rate = 0.1
        self.learning_rate_decay_factor = 0.99
        # Training is stopped when the learning rate goes under this threshold.
        self.learning_rate_threshold = 1.0e-5
        # If not None, clip gradients to this value.
        self.clip_gradients = 5.0
        self.num_epochs = 20