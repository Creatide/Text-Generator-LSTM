# Text Generator LSTM

![Training Process Example](https://i.imgur.com/8M5pIty.gif)

The Text Generator LSTM is a Python script that uses Keras' LSTM to train and generate new text based on the style of the provided reference texts.

## About

The main objective of this project was to study text generation and experiment with developing a simple script that generates lyric ideas based on references in training. However, it is not limited to just lyrics, as it can also be used for other types of text. Moreover, the project aimed to design a "user-friendly" script that efficiently produces text files based on specified parameters through terminal commands. 

In addition, the script provides the flexibility to select custom parameters such as genre (if an info.json file is included in the folder) or artist/text (based on the folder name) for training. This is accomplished through scanning the folders and utilizing JSON objects.

The script also includes a spell checking feature, which utilizes the ['pyspellchecking'](https://pyspellchecker.readthedocs.io/en/latest/) module to attempt to correct nonsensical words in the generated text.

### Example of Generated Results

Trained model from Black Metal band lyrics.
```
# Model: Optimizer: Adam, Corpus Length: 10657199,  loss: 1.2125, accuracy: 0.6170, val_loss: 1.3453, val_accuracy: 0.5879

A moment that the gods pretend.
The sea of hate and procession.
Where story you are.
I am the fire devoured to seek the storm in the eternal cold.
The neck of death.
Like truths that extinction.
Mighty waters close.
Hallowed in the trees of stone.
In the hour of my mind.
The man is not the sign.
In the eyes of hope shall be the new purge.
And the story bloody breath of trial.

# Model: Optimizer: Adam, Corpus Length: 10657196, loss: 1.3256, accuracy: 0.5849, val_loss: 1.3567, val_accuracy: 0.5782

The stones.
Everything is so lead the steps.
Silence that the worlds and an everything.
My eyes before the dim of the fire.
And so thou had waited the lonely moon.
The sins shall found the soul.
The earth is on the eye of your cross.
This is the anger of the soul we hate.
In the night of the same.
For you will never complete of the voices.
The way of war burns from the way.

# Model: Optimizer: RMSprop, Corpus Length: 8888980, loss: 1.3416, accuracy: 0.5864, val_loss: 1.4301, val_accuracy: 0.5665

Dark thing where the souls of the end.
The dark of pain is planet.
Which fly to the winds are passed by shadows.
Will lose the flames is in the serpent.
More than the destiny bodies and i am return.
The warriors of the dark.
And i will come to musical lives.
Have the light of death.
To die to the corruption.
Stripped in the past.
Infinite for eternal trees.
When the gates of secrets to be flee.
```
# Install

## Requirements

I've been running this script on the following setup:
* Windows 11 
* Ubuntu 22.10 (via WSL)
* Python: 3.10.9
* Numpy: 1.24.2
* Keras: 2.11.0

## Install Commands

To run Keras using a GPU, follow the instructions 2A. section for GPU installation. However, if you wish to use a CPU, use the CPU commands from section 2B. instead.

**1. (Optional, Windows) Install WSL, Ubuntu and Anaconda**
```
wsl --install
wsl --install Ubuntu
wsl -d ubuntu
cd ~
wget https://repo.continuum.io/archive/Anaconda3-2022.10-Linux-x86_64.sh
bash Anaconda3-2022.10-Linux-x86_64.sh
```
**2A. (GPU) Install Anaconda and [TensorFlow](https://www.tensorflow.org/install/pip#linux_setup)**
```
conda create -n tf python=3.10 pip
conda activate tf
conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
python3 -m pip install tensorflow
conda install -c nvidia cuda-nvcc=11.3.58
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
printf 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/\nexport XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/\n' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice
cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/
```
**2B. (CPU) Install Anaconda and [TensorFlow](https://www.tensorflow.org/install/pip#linux_setup)**
```
conda create -n tf python=3.10 pip tensorflow
```
**3. Git clone Text Generator LSTM repository**
```
git clone https://github.com/Creatide/Text-Generator-LSTM.git
```
**4. Use Text Generator LSTM**
Now it should workd by running commands using terminal
```
python main.py <COMMANDS>
```
## Spell Checking Module

To use Python Spell Checking feature, you need to install ['pyspellchecker'](https://pyspellchecker.readthedocs.io/en/latest/) to make it to work.
```
pip install pyspellchecker
```

# Commands

![Generate Process Example](https://i.imgur.com/Aio64f6.gif)

You can use this script with terminal commands. Here is couple examples how to use those:

## Train Dracula Example
Here's an example of how to train a model and generate text using the Dracula book provided in the `data` folder for testing purposes.
```
# Train model based Dracula book
python main.py -t -m dracula -i Dracula

# Generate text using model
python main.py -g -m dracula -c 10
```

## How Command Arguments Works?
Train `-t` model and save it to folder `data/models` using name `-m MyModelName`. Then we use only these artists and custom genre for training data using `-i`.
```
python main.py -t -m MyModelName -i "Alice in Chains, Nirvana, grunge"
```

Generate `-g` texts using model named `-m MyModelName`. Generated text length `-l` will be 400 characters and we will generate 10 pieces `-c` of different texts.
```
python main.py -g -m MyModelName -l 400 -c 10
```


## Train Commands

| Command             | Arguments            | Description                                                  |
| ------------------- | -------------------- | ------------------------------------------------------------ |
| **-t** / --train    | -               | Train                                                        |
| **-e** / --evaluate | -               | Evaluate Training mode. This mode is for testing how training evaluates while it shows results with different temperatures after each epoch run. |
| **-i** / --items    | "Nirvana, Metallica" | List of items that wanted to use for training                |
| **-m** / --model    | "ModelName"          | Model name that will be used for saving model in training process. Extension not needed for default .h5 models. |

## Generate Commands

| Command             | Arguments     | Description                                                  |
| ------------------- | ------------- | ------------------------------------------------------------ |
| **-g** / --generate | -        | Generate text using trained models. You need to train or have existing model, in "data/models" folder. |
| **-m** / --model    | "ModelName"   | Model that wanted to use in generation process.              |
| **-p** / --primer   | "Hello World" | Primer text for generating process. Every generated text begins with this and neural network continue based on that. |
| **-l** / --length   | Integer       | Length of generated text in characters.                      |
| **-c** / --count    | Integer       | Count of how many texts will be generated. If text "MERGE_RESULTS" is on, then it will be text blocks on that generated document. |

## General Commands

| Command            | Arguments                             | Description                                              |
| ------------------ | ------------------------------------- | -------------------------------------------------------- |
| **-o** / --options | e.g.: "epochs=10, learning_rate=0.01" | All possible arguments that is in use in main.py script. |
| **-v** / --version | -                                | Show versions of installed python, numpy and keras.      |

## Constants

The 'constants.py' file contains a wide range of options for both training and generating. Each option is explained in the file, and adjusting them can lead to better training results. Keep in mind that **any values set in 'constants.py' can be overwritten by command line arguments**, if necessary. By experimenting with different settings, you can optimize your training process and achieve even better results.

```
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
PRIMER_TEXT = ''
GENERATE_TEXT_LENGTH = 400
GENERATE_TEXTS_COUNT = 1
TEMPERATURE = 0.6
MERGE_RESULTS = True
TEXT_FORMATTING = True
USE_SPELLCHECKER = True
SEQUENCE_LENGTH = 32
STEP_SIZE = 3
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 0.01
LSTM_LAYERS = [256, 256]
DROPOUT_LAYERS = 0
DENSE_LAYERS = ['linear']
LOSS_FUNCTION = 'categorical_crossentropy'
ACTIVATION_LAYER = 'softmax'
OPTIMIZER = 'adam'
USE_CHECKPOINTS = True
STEPS_PER_EPOCH = None
SHUFFLE_DATA = True
USE_TENSORBOARD = True
USE_VALIDATION = True
VALIDATION_SPLIT = 0.2
REDUCE_LR_STUCK_FACTOR = 0.5
EVALUATE_TEMPERATURES = [0.5, 0.8, 1.0]
USE_EARLY_STOPPING = False
RESTORE_BEST_WEIGHTS = True
TRAIN_PATIENCE = 6
MONITOR_METRIC = 'val_loss'
```
**Some "Rule of Thumb" notes that I've gathered while studying about training parameters:**

* **SEQUENCE_LENGTH**: The length of the input sequences used for training the model. This should be set to a value that makes sense for your data. In general, shorter sequences will make the model train faster but may result in lower accuracy, while longer sequences may take longer to train but may result in higher accuracy. A good rule of thumb is to start with a sequence length of 64-128 and adjust as needed.

* **STEP_SIZE**: The step size between the sequences used for training the model. This should be set to a value that makes sense for your data. A smaller step size will result in more sequences and may improve accuracy, but will also increase training time. A good rule of thumb is to start with a step size of 2-5.

* **BATCH_SIZE**: The number of training examples used in one forward/backward pass of the model. This value should be set based on the available memory on your machine. A larger batch size can result in faster training, but may require more memory. A good rule of thumb is to start with a batch size of 32-128.

* **EPOCHS**: The number of iterations over the entire training data set. This value depends on the complexity of your data and the desired level of accuracy. In general, a larger number of epochs will result in higher accuracy, but may also increase the risk of overfitting. A good rule of thumb is to start with 10–20 epochs and adjust as needed.

* **LEARNING_RATE**: The rate at which the model updates its weights during training. This value should be set based on the complexity of your data and the desired level of accuracy. A higher learning rate can result in faster convergence, but may also result in overshooting the optimal weights. A good rule of thumb is to start with a learning rate of 0.001 and adjust as needed.

* **LSTM_LAYERS**: The number and size of LSTM layers in the model. This value depends on the complexity of your data and the desired level of accuracy. In general, a larger number of layers and/or larger layer sizes can result in higher accuracy, but may also increase the risk of overfitting. A good rule of thumb is to start with 1–2 layers with 64–256 units each and adjust as needed.

* **DROPOUT_LAYERS**: The dropout rate to use to prevent overfitting. This value should be set based on the risk of overfitting in your model. A higher dropout rate can result in less overfitting, but may also reduce accuracy. A good rule of thumb is to start with a dropout rate of 0.2 and adjust as needed.

* **DENSE_LAYERS**: The number and size of dense layers in the model. This value depends on the complexity of your data and the desired level of accuracy. In general, adding dense layers can improve accuracy, but may also increase the risk of overfitting. A good rule of thumb is to start with 1-2 and start with a small number of units, such as 64 or 128, and gradually increase the number if necessary. Adding more units can increase the capacity of the model and potentially improve accuracy, but may also increase the risk of overfitting.

* **LOSS_FUNCTION**: The quantity that the model should seek to minimize during training. This value should be set based on the type of problem you are trying to solve. A good rule of thumb is to use `categorical_crossentropy` for multi-class classification problems.

* **ACTIVATION_LAYER**: The activation function to use on the output layer. This value should be set based on the type of problem you are trying to solve. A good rule of thumb is to use `softmax` for multi-class classification problems.

* **OPTIMIZER**: The optimization algorithms:
  * `Adam`: The Adam optimizer is a good general-purpose optimizer that works well for many types of models and datasets. It's a popular choice and generally a safe bet.
  * `RMSprop`: If Adam is causing convergence issues or instability, RMSprop is a good alternative. It's less aggressive than Adam and can be more stable, especially for recurrent neural networks like LSTMs.
  * `Stochastic Gradient Descent` (SGD) with momentum: If you have a lot of data and a lot of parameters, stochastic gradient descent (SGD) with momentum can work well. It's a bit slower than Adam and RMSprop, but can converge well with large amounts of data.
  * Tune the learning rate: No matter which optimizer you choose, it's important to tune the learning rate for your specific problem. A learning rate that's too high can cause the optimizer to diverge, while a learning rate that's too low can cause slow convergence.
  
* **USE_VALIDATION**: A portion of the training data will be held out and used for validation to assess the model's performance on unseen data. This helps to determine the model's suitability for deployment. By default, this is set to True.

* **VALIDATION_SPLIT**: The proportion of training data to use for validation. If this parameter is set to 0.2, then 20% of the training data will be used for validation, and the remaining 80% will be used for training. A good rule of thumb is to set this to 0.1-0.3.

* **REDUCE_LR_STUCK_FACTOR**: A parameter that controls the learning rate reduction strategy. When the model's performance on the validation set stops improving, the learning rate will be reduced by this factor (e.g., if this parameter is set to 0.5, then the learning rate will be reduced by half). A good rule of thumb is to set this to 0.1-0.5.

# Use TensorBoard

To use TensorBoard, simply set the `USE_TENSORBOARD` flag to `True` in the `constants.py` file. This will save metrics data from the training process to a `debug` folder, which you can then visualize in TensorBoard. To launch TensorBoard and view your training progress, navigate to the project root directory and run the following command:
```
tensorboard --logdir=./debug
```
![TensorBoard UI](https://i.imgur.com/VieOV7F.png)

> If you want to view all diagrams simultaneously on a single page, simply use the "/*" mark in the filter field.

# Data Crawling & Links
So now you have Text Generator running and you need data for training? Here is couple tips and scripts that can help you to collect some lyrics and download free books.

## Data Crawling

* [LyricsGenius](https://lyricsgenius.readthedocs.io/en/master/): a Python client for the Genius.com API.
  * Great tool to get all lyrics of chosen artist. 
  * There is good [guide how to use it easily](https://lyricsgenius.readthedocs.io/en/master/setup.html).
  * I created a [script](https://pastebin.com/kRsaLeJU) that prepares downloaded lyrics to .txt format and cleans un-wanted words and symbols from it.
  * [Same script as above](https://pastebin.com/iseCK2iF), but with language and duplicates detection. Needed it for saving only English language versions from lyrics.

* [Azapi](https://github.com/elmoiv/azapi): A fast and secure API for AZLyrics.com to get lyrics easily.
  * Another tool to download lyrics, but from AZLyrics.com.
  * I created a [script](https://pastebin.com/YPLRB2Wz) that downloads all of chosen artist lyrics.
  * Unfortunately, AZLyrics.com may sometimes ban your IP if you download too many lyrics from there, (that's the reason why there is randomized wait times in the script).

## Free Books

* [Project Gutenberg](https://www.gutenberg.org/) is a library of over 60,000 free eBooks. There you can download books in .txt format also for training.

# Want to Help?

I am new to AI and have created a script for practice. However, as I lack experience in this field, the output generated by the script may not make sense. If you have any suggestions on how I can improve the script's training, please feel free to contribute by making a pull request. Your help is greatly appreciated.

# Updated

* **19.02.2023**
  * Added support for [spellchecker](https://pypi.org/project/pyspellchecker). By default it's work for english, but there is also support for English - ‘en’, Spanish - ‘es’, French - ‘fr’, Portuguese - ‘pt’, German - ‘de’, Russian - ‘ru’, Arabic - ‘ar’.
  * Spellchecking checks generated words and try to get the one "most likely" answer. Tested and seems to work for my lyrics quite nicely.
  * If english frequency list is not good, there could be possible to add new list like this one: https://github.com/dwyl/english-words

# Licence

MIT License

Copyright (c) 2023 Creatide

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
