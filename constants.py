# ================================================================================================ #
# GENERAL                                                                                          #
# ================================================================================================ #

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

# ================================================================================================ #
# GENERATE PROCESS                                                                                 #
# ================================================================================================ #

# Starting text for generating new text that will continue from this.
PRIMER_TEXT = ''
# Length of the generated text in sentences.
GENERATE_TEXT_LENGTH = 400
# Setting determines how many texts files the model will generate.
GENERATE_TEXTS_COUNT = 1
# Level of randomness in the generated output. "Higher = creative, lower = safe pick".
TEMPERATURE = 0.8
# Export all generated lyrics to same text file per generation process.
MERGE_RESULTS = True
# Stylize text output to more readable format e.g. separate sentences to new lines.
TEXT_FORMATTING = True
# Use spellchecker for English by default. Supported languages: https://pyspellchecker.readthedocs.io
USE_SPELLCHECKER = True

# ================================================================================================ #
# TRAINING PROCESS                                                                                 #
# ================================================================================================ #

# Hyperparameters -------------------------------------------------------------------------------- #

# The length of the input sequences used for training the model.
# High value could cause 'Dst tensor is not initialized' https://stackoverflow.com/a/40389498/1629596
SEQUENCE_LENGTH = 128
# The step size between the sequences used for training the model.
STEP_SIZE = 3
# The number of training examples used in one forward/backward pass of the model.
BATCH_SIZE = 256
# The number of iterations over the entire training data set.
EPOCHS = 1000
# The rate at which the model updates its weights during training.
LEARNING_RATE = 0.001
# The number is unit size of each LSTM layer of the model.
# Set multiple LSTM layers to model by using list, e.g. three different size of LSTM layers [256, 128]
# Note that you need at least one LSTM layer to create model
LSTM_LAYERS = [256, 256]
# Dropout prevent overfitting. It randomly selected neurons are ignored during training.
# Add dropout layer after every LSTM layer with chosen value between 0.0-1.0 (0.2 is 20%)
# Disable dropout layers byt settings it to 0.
DROPOUT_LAYERS = 0.3
# Convolutional is used to perform 1-dimensional convolutional operations on sequential data, such as
# time series or text data, and is commonly used in tasks such as feature extraction and sequence classification.
# e.g. [[64, 8], [128, 4]] will create two conv1D layers. First value is filter size and second is kernel size.
# e.g. If want to create only one layer, just use format [64, 8]
CONVOLUTIONAL_LAYERS = [32, 6], [64, 3]
# You can se multiple dense layers by giving multiple activation functions e.g. ['linear', 'relu']
# Leave list to empty if you want to disable dense layers
# Keras default dense layer activation is 'linear', other commonly used is 'relu', 'sigmoid', 'softmax'
# List of all available activations in keras https://keras.io/api/layers/activations/
# e.g.: 'linear', 'relu', 'softmax'
DENSE_LAYERS = ['relu', 'relu']
# The embedding layer in deep learning maps the input data into a lower-dimensional vector space where
# similar words or features are closer to each other.
# e.g.: 32, 0 = disabled
EMBEDDING_LAYER = 128
# L1 and L2 regularizers are prevent overfitting. They adding a penalty term to the loss function.
# L1: Encourages the model to learn sparse patterns, where many of the weights are zero.
# L2: Encourages the model to learn smaller weights overall, which can help to prevent overfitting.
# L2 regularization is also known as weight decay, because it causes the weights to decay towards zero over time.
REGULARIZER_L1 = 0.00001
REGULARIZER_L2 = 0.00001
# Computes the crossentropy loss between the labels and predictions.
# https://keras.io/api/losses/, https://neptune.ai/blog/keras-loss-functions
# Example: categorical_crossentropy, binary_crossentropy
LOSS_FUNCTION = 'categorical_crossentropy'
# Applies an activation function to an output.
ACTIVATION_LAYER = 'softmax'
# Adam: Stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
# RMSprop: Moving average of the square of gradients. Divide the gradient by the root of this average.
OPTIMIZER = 'adam'
# Save checkpoints while training.
USE_CHECKPOINTS = True
# Total number of steps before declaring one epoch finished. By default value is: None
# Common approach is to divide the number of training samples by the batch size.
# 10000 training samples and a batch size of 32, you would set steps_per_epoch to 10000/32 = 313.
STEPS_PER_EPOCH = None
# Shuffle the training data before each epoch to prevent the model from learning the order of the data.
# Has no effect when steps_per_epoch is not None.
SHUFFLE_DATA = True

# Debugging -------------------------------------------------------------------------------------- #

# Save tensorboard training debug data to debug folder
USE_TENSORBOARD = True
# Validation process helps to determine the model's suitability for deployment.
USE_VALIDATION = True
# Split training data into two parts, one for training and one for evaluating performance.
# For example, if the value is 0.2, then 20% of the training data will be used for validation.
VALIDATION_SPLIT = 0.2
# Reduce learning rate when a metric has stopped improving. (new_lr = lr * factor.)
# Set None or 0 to disable this feature
REDUCE_LR_STUCK_FACTOR = 0.2
# Perplexity measures how well a model predicts the next word in a sequence, given the previous words. A lower perplexity
# score indicates that the model is better at predicting the next word, while a higher perplexity score indicates that the
# model is less accurate. This value is only for debugging to see perplexity value.
# Note: If you train your model using perplexity and continue training later on, you must ensure that this is enabled!
USE_PERPLEXITY_METRIC = True
# Evaluate training mode temperatures that will be used for generation phase
# This affects only to generated example texts temperatures while in evaluate mode
EVALUATE_TEMPERATURES = [0.5, 0.8, 1.0]

# Early Stopping --------------------------------------------------------------------------------- #

# Stop training when a monitored metric has stopped improving.
USE_EARLY_STOPPING = True
# During training, the EarlyStopping callback continuously monitors the quantity
# Training process stops when it detects that the monitored quantity has stopped improving.
# e.g.: 'loss', 'accuracy', 'val_loss', 'val_accuracy',
EARLY_STOPPING_MONITOR = 'accuracy'
# Restore model weights from the epoch with the best value of the monitored quantity.
# If False, the model weights obtained at the last step of training are used.
RESTORE_BEST_WEIGHTS = True
# Number of epochs with no improvement after which training will be stopped if early stopping is activated.
TRAIN_PATIENCE = 10
# EarlyStopping with min_delta monitors the change in the monitored metric and stops the training process
# when the change is less than the specified min_delta value. Default is 0
EARLY_STOP_MIN_DELTA = 0.001
# Metric for monitoring the training progress of a model during training. The metric is evaluated on the
# validation set after each epoch and can be used to determine when to stop training if the performance on
# the validation set stops improving.
# e.g.: 'accuracy'
MONITOR_METRIC = 'accuracy'