# Text Generator LSTM

![Training Process Example](https://i.imgur.com/8M5pIty.gif)

The Text Generator LSTM is a Python script that uses Keras' LSTM to train and generate new text based on the style of the provided reference texts.

## About

The main objective of this project was to study text generation and experiment with developing a simple script that generates lyric ideas based on references in training. However, it is not limited to just lyrics, as it can also be used for other types of text. Moreover, the project aimed to design a "user-friendly" script that efficiently produces text files based on specified parameters through terminal commands. 

In addition, the script provides the flexibility to select custom parameters such as genre (if an info.json file is included in the folder) or artist/text (based on the folder name) for training. This is accomplished through scanning the folders and utilizing JSON objects.

# Install

**1. (Optional, Windows) Install WSL, Ubuntu and Anaconda**
```
wsl --install
wsl --install Ubuntu
wsl -d ubuntu
cd ~
wget https://repo.continuum.io/archive/Anaconda3-2022.10-Linux-x86_64.sh
```
**2. (GPU) Install Anaconda and [TensorFlow](https://www.tensorflow.org/install/pip#linux_setup)**
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
**2. (CPU) Install Anaconda and [TensorFlow](https://www.tensorflow.org/install/pip#linux_setup)**
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

# Commands

![Generate Process Example](https://i.imgur.com/Aio64f6.gif)

You can use this script with terminal commands. Here is couple examples how to use those:

Train (-t) model and save it to folder "data/models" using name -(m) MyModelName.h5. Then we use only these artists and custom genre for training data (-i).
```
python main.py -t -m MyModelName -i "Alice in Chains, Nirvana, grunge"
```

Generate texts (-g) using model named (-m) MyModelName. Generated text length (-l) will be 400 characters and we will generate 10 pieces (-c) of different texts.
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



# Data Crawling Tips
So now you have Text Generator running and you need data for training? Here is couple tips and scripts that can help you to collect some lyrics and download free books.

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
