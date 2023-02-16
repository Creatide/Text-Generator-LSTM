# Text Generator LSTM

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

# Usage

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
