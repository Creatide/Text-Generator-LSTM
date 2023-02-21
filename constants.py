# NAMING
# ------------------------------------------------------------------------------------------------ #
PATH_DATA_FOLDER = 'data'
PATH_TEXTS_FOLDER = 'texts'
PATH_RESULTS_FOLDER = 'results'
PATH_MODELS_FOLDER = 'models'
PATH_CHECKPOINTS_FOLDER = 'checkpoints'
PATH_DEBUG_FOLDER = 'debug'
JSON_INFO_FILENAME = 'info.json'
DEFAULT_MODEL_NAME = 'default'
DEFAULT_MODEL_FORMAT = '.h5'
DEFAULT_MODEL_FILENAME = DEFAULT_MODEL_NAME + DEFAULT_MODEL_FORMAT

# GENERATE
# ------------------------------------------------------------------------------------------------ #
# Starting text for generating new text that will continue from this.
PRIMER_TEXT = ''
# Length of the generated text in sentences.
GENERATE_TEXT_LENGTH = 400
# Setting determines how many texts files the model will generate.
GENERATE_TEXTS_COUNT = 1
# Export all generated lyrics to same text file per generation process.
MERGE_RESULTS = True
# Stylize text output to more readable format e.g. separate sentences to new lines.
TEXT_FORMATTING = True
# Level of randomness in the generated output. "Higher = creative, lower = safe pick".
TEMPERATURE = 0.6
# Use spellchecker for English by default. Supported languages: https://pyspellchecker.readthedocs.io
USE_SPELLCHECKER = True

# TRAINING
# ------------------------------------------------------------------------------------------------ #
# The length of the input sequences used for training the model.
# High value could cause 'Dst tensor is not initialized' https://stackoverflow.com/a/40389498/1629596
SEQUENCE_LENGTH = 32
# The step size between the sequences used for training the model.
STEP_SIZE = 3
# The number of training examples used in one forward/backward pass of the model.
BATCH_SIZE = 128
# The number of iterations over the entire training data set.
EPOCHS = 1000
# The rate at which the model updates its weights during training.
LEARNING_RATE = 0.01
# The number of units in each LSTM layer of the model.
LSTM_UNITS = 256
# Save checkpoints while training.
USE_CHECKPOINTS = True
# Validation process helps to determine the model's suitability for deployment.
USE_VALIDATION = True
# Split training data into two parts, one for training and one for evaluating performance.
# For example, if the value is 0.2, then 20% of the training data will be used for validation.
VALIDATION_SPLIT = 0.2
# Number of epochs with no improvement after which training will be stopped.
TRAIN_PATIENCE = 6
# Restore model weights from the epoch with the best value of the monitored quantity. 
# If False, the model weights obtained at the last step of training are used. 
RESTORE_BEST_WEIGHTS = True
# Reduce learning rate when a metric has stopped improving. (new_lr = lr * factor.)
# Set None or 0 to disable this feature
REDUCE_LR_STUCK_FACTOR = 0.5
# Shuffle the training data before each epoch to prevent the model from learning the order of the data.
# Has no effect when steps_per_epoch is not None.
SHUFFLE_DATA = True
# Total number of steps before declaring one epoch finished. By default value is: None
# Common approach is to divide the number of training samples by the batch size. 
# 10000 training samples and a batch size of 32, you would set steps_per_epoch to 10000/32 = 313.
STEPS_PER_EPOCH = None
# During training, the EarlyStopping callback continuously monitors the quantity
# Training process stops when it detects that the monitored quantity has stopped improving.
# e.g.: val_loss, loss, accuracy, precision
MONITOR_METRIC = 'val_loss'
# Compute the quantity that a model should seek to minimize during training.
# More: https://keras.io/api/losses/, https://neptune.ai/blog/keras-loss-functions
# Example: categorical_crossentropy, binary_crossentropy
LOSS_FUNCTION = 'categorical_crossentropy'
# Applies an activation function to an output.
ACTIVATION_LAYER = 'softmax'
# Adam: Stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
# RMSprop: Moving average of the square of gradients. Divide the gradient by the root of this average.
OPTIMIZER = 'adam'
# Save tensorboard training debug data to debug folder
# https://www.tensorflow.org/tensorboard/get_started
USE_TENSORBOARD = False
# Evaluate training mode temperatures that will be used for generation phase
EVALUATE_TEMPERATURES = [0.5, 0.8, 1.0]
