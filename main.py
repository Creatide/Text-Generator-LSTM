'''
MIT License

Copyright (c) Creatide.com (Sakari Niittymaa)

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
'''

import os
import sys
import datetime
import getopt
import json
import random
import numpy as np
from pathlib import Path

from tensorflow import keras
from keras import layers

import constants as c

class TextGenerator:
    
    # ============================================================================================ #
    # INITIALIZE                                                                                   #
    # ============================================================================================ #
    
    def __init__(self, arguments):
        
        self.arguments = arguments
        self.model = None
        self.model_history = None
        self.model_filename = None
        self.model_found = False
        self.text = None
        self.primer = ''        
        self.characters = ''
        self.allowed_characters = 'abcdefghijklmnopqrstuvwxyzåäö-,.!?\'" 0123456789'
        self.character_indices = None
        self.indices_character = None
        self.next_characters = []
        self.x = self.x_validation = None
        self.y = self.y_validation = None
        self.texts_length = int(self.arguments['texts_length']) if isinstance(int(self.arguments['texts_length']), int) else c.GENERATE_TEXT_LENGTH
        self.texts_count = int(self.arguments['texts_count']) if isinstance(int(self.arguments['texts_count']), int) else c.GENERATE_TEXTS_COUNT
        self.process_id = datetime.datetime.now().strftime('%m%d%Y%H%M%S%f')
        self.debug = self.arguments['debug']

        self.data_folder_path = self.get_data_folder_path()
        self.data_folder_json = self.cra_data_folder_json()
        self.set_model()
        
        if self.arguments['options']: self.set_options()
        if self.arguments['generate']: self.generate() 
        if self.arguments['train']: self.train()         
        if self.arguments['evaluate']: self.evaluate_training()         
    
    
    # ============================================================================================ #
    # GENERATE TEXT PROCESS                                                                        #
    # ============================================================================================ #
    
    # Generate Text ------------------------------------------------------------------------------ #
    def generate(self):
        # Dont load models while generating text in other modes
        if self.arguments['generate']:
            self.load_model_file()     
            self.prepare_data()
        
        # Create initial sentence from corpus random location by selecting index
        start_index = random.randint(0, len(self.text) - self.arguments['sequence_length'] - 1)
        # Slice sentence from corpus
        sentence = self.text[start_index : start_index + self.arguments['sequence_length']]
        # Set initial text or randomly picked from corpus
        generated_text = self.primer if len(self.primer) else sentence
        
        # Generate text process
        for i in range(self.texts_length):
            
            x_predictions = np.zeros((1, self.arguments['sequence_length'], len(self.characters)))
            
            for t, character in enumerate(sentence):
                x_predictions[0, t, self.character_indices[character]] = 1.0
                
            predictions = self.model.predict(x_predictions, verbose=0)[0]
            next_index = self.sample(predictions, self.arguments['temperature'])
            next_character = self.indices_character[next_index]
            sentence = sentence[1:] + next_character
            generated_text += next_character
            
            # Print progresss percent
            if not self.arguments['evaluate'] and i % 20 == 0:
                print('Process:', round(i / self.texts_length * 100),"%",)
        
        # Remove randomly picked text from beginning if primal text not found
        if len(self.primer) < 1: generated_text = generated_text[len(sentence):]
        
        # Use text formatting
        if self.arguments['text_formatting']:
            generated_text = self.format_text(generated_text)
            
        # Generated result in generate process
        if self.arguments['generate']:
            if self.texts_count > 0:
                print(f"\nGeneration Process Remaining: {self.texts_count - 1} of {self.arguments['texts_count']}\n")
                print(generated_text, '\n')
                
                if self.arguments['merge_results']:
                    self.save_text_file(generated_text, name=self.process_id, path=c.PATH_RESULTS_FOLDER, merge_results=self.arguments['merge_results'])
                else:
                    self.save_text_file(generated_text, path=c.PATH_RESULTS_FOLDER, merge_results=self.arguments['merge_results'])
                    
                # Kill last process to prevent useless preparation run
                self.texts_count -= 1
                if self.texts_count == 0:
                    exit()
                self.generate()
                
        return generated_text
    
    
    # ============================================================================================ #
    # TRAIN PROCESS                                                                                #
    # ============================================================================================ #

    # Train Model -------------------------------------------------------------------------------- #
    def train(self) -> None:        
        # Print training arguments
        print('\nTRAIN ARGUMENTS:')
        for k, v in self.arguments.items():
            print(k + ':', v)
        
        self.prepare_data()
        # If model with same name found, load it
        if self.model_found: 
            self.load_model_file() 
        else: 
            self.model_build()                                          

        # https://keras.io/api/models/model_training_apis/
        self.model_fit(self.arguments['epochs'])        
        self.save_model_file(self.arguments['model']['filename'])


    # Evaluate Training Mode --------------------------------------------------------------------- #
    def evaluate_training(self) -> None:        
        # Print training arguments
        print_output = '\nTRAIN ARGUMENTS:\n'
        for k, v in self.arguments.items():
            print_output += str(k) + ':' + str(v) + '\n'
        print(print_output) 
                       
        self.save_text_file(print_output, path=c.PATH_RESULTS_FOLDER, name="evaluate_training", merge_results=True)
        
        self.prepare_data()
        
        # If model with same name found, load it
        if self.model_found: 
            self.load_model_file()
        else: 
            self.model_build()
        
        for epoch in range(self.arguments['epochs']):
            
            self.model_fit()
            
            print_output = '\n***** EPOCH CHANGE *****\n\n'
            
            for temperature in c.EVALUATE_TEMPERATURES:
                
                print_output += 'Temperature: ' + str(temperature) + ', Epoch: ' + str(epoch+1) + '/' + str(self.arguments['epochs']) + '\n'
                print_output += str(self.model_history.history) + '\n'
                print_output += '-'*len(str(self.model_history.history)) + '\n'
                print_output += self.generate() + '\n'
                
                print(print_output)                
                self.save_text_file(print_output, path=c.PATH_RESULTS_FOLDER, name="evaluate_training", merge_results=True)
                self.save_model_file(self.arguments['model']['filename'])
                
                print_output = ''                
                self.arguments['temperature'] = temperature
                            

    # ============================================================================================ #
    # PREPARE & POST PROCESS                                                                       #
    # ============================================================================================ #    
    
    # Prepare Data For Training ------------------------------------------------------------------ #
    def prepare_data(self):
        
        print('TEXT INFO:')
        
        if self.arguments['primer']:
            self.primer = self.clean_string(self.arguments['primer'])
            print('Primer Text:', self.primer)
        
        self.text = self.get_texts(self.arguments['items'])
        print('Corpus Length:', len(self.text))
        
        # Get text characters and compare to allowed_characters
        # self.characters = sorted(set(self.text))
        # if len(self.characters) < 2:
        #     print('Text must contain at least 2 distinct characters')
        #     exit()
        # elif not all(char in self.allowed_characters for char in self.characters):
        #     invalid_characters = [char for char in self.characters if char not in self.allowed_characters]
        #     print(f'ERROR: Text contains invalid characters: {", ".join(invalid_characters)}')
        #     exit()
        # else:
        # Use same character always to avoid list size errors
        self.characters = sorted([*self.allowed_characters])
        
        print('Total Characters:', len(self.characters))
        print('Allowed Characters:', ''.join(self.characters))
        
        # Create indicies for characters
        self.character_indices = dict((c, i) for i, c in enumerate(self.characters))
        self.indices_character = dict((i, c) for i, c in enumerate(self.characters))
        
        sentences = []
        self.next_characters = []
        for i in range(0, len(self.text) - self.arguments['sequence_length'], self.arguments['step_size']):
            sentences.append(self.text[i : i + self.arguments['sequence_length']])
            self.next_characters.append(self.text[i + self.arguments['sequence_length']])
            
        print(f"Number of Sequences: {len(sentences)}\n")
        
        self.x = np.zeros((len(sentences), self.arguments['sequence_length'], len(self.characters)), dtype=bool)
        self.y = np.zeros((len(sentences), len(self.characters)), dtype=bool)
        
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                self.x[i, t, self.character_indices[char]] = 1
            self.y[i, self.character_indices[self.next_characters[i]]] = 1
            
        # Reserve samples for validation
        if self.arguments['validation']:
            split_index = int(len(self.x) * self.arguments['validation_split'])
            self.x_validation = self.x[-split_index:]
            self.y_validation = self.y[-split_index:]
            self.x = self.x[:-split_index]
            self.y = self.y[:-split_index]
            
    
    # Build The Model: A Single LSTM Layer ------------------------------------------------------- #
    def model_build(self):
        self.model = keras.Sequential(
            [
                keras.Input(shape=(self.arguments['sequence_length'], len(self.characters))),
                layers.LSTM(self.arguments['lstm_units']),
                # layers.Dense(units=32, activation=self.arguments['activation_layer']),
                layers.Dense(len(self.characters), activation=self.arguments['activation_layer']),
            ]
        )
        optimizer = keras.optimizers.RMSprop(learning_rate=self.arguments['learning_rate'])
        self.model.compile(loss=self.arguments['loss_function'], optimizer=optimizer, metrics=['accuracy'])
    
    
    # Build The Model: Fit Model Function -------------------------------------------------------- #
    def model_fit(self, epochs=1) -> None:
        
        self.model_history = self.model.fit(
            x=self.x, 
            y=self.y, 
            batch_size=self.arguments['batch_size'], 
            epochs=epochs, 
            callbacks=[self.create_model_callbacks()],
            validation_data=(self.x_validation, self.y_validation) if self.arguments['validation'] else None,
            validation_split=self.arguments['validation_split'] if not self.arguments['validation'] else None,
            steps_per_epoch=self.arguments['steps_per_epoch'] if self.arguments['steps_per_epoch'] else None,
            shuffle=self.arguments['shuffle_data'],            
            )
        
    
    # Create Callbacks For Model ----------------------------------------------------------------- #
    def create_model_callbacks(self):
        # Create debug folder if needed
        if not os.path.exists(c.PATH_DEBUG_FOLDER): os.makedirs(c.PATH_DEBUG_FOLDER)
        # Create filepath for checkpoints
        checkpoint_filepath = os.path.join(self.data_folder_path, c.PATH_MODELS_FOLDER, c.PATH_CHECKPOINTS_FOLDER, self.arguments['model']['name'])
        
        # https://keras.io/api/callbacks/
        model_callback = [
            keras.callbacks.EarlyStopping(monitor=self.arguments['monitor_metric'], patience=self.arguments['train_patience'], verbose=1, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=round(self.arguments['train_patience'] / 2), min_lr=0.001),
            keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_filepath) + '_{epoch:02d}-{loss:.2f}' + self.arguments['model']['extension'], save_best_only=True),
            
        ]
        
        # https://www.tensorflow.org/tensorboard/get_started
        if self.arguments['debug']:
            model_callback.append(keras.callbacks.TensorBoard(log_dir='./' + c.PATH_DEBUG_FOLDER))
        
        return model_callback
    
    
    # Prepare The Text Sampling Function --------------------------------------------------------- #
    def sample(self, preds, temperature=c.TEMPERATURE):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    
    
    # Get Texts From Data For Traning and Generating Random Primer Text Sample ------------------- #
    def get_texts(self, *args):
        authors_found = 0
        texts_found = 0
        texts_string = ''        
        author_data = self.json_search_objects(self.data_folder_json, ['name', 'genre'], args)
        
        # Get all required texts based on file locations in json
        for author in author_data:
            authors_found += 1
            for text in author['children']:
                if text['type'] == 'txt':
                    with open(os.path.join(self.data_folder_path, c.PATH_TEXTS_FOLDER, author['name'], text['name'])) as texts:
                        texts_string += texts.read()
                        texts_found += 1
                        
        cleaned_string = self.clean_string(texts_string)     
        
        if not self.arguments['evaluate']:
            print('Authors Found:', authors_found)   
            print('Texts Found:', texts_found)
            print('Texts Length:', len(cleaned_string))
        
        return cleaned_string
    
    
    # Set Model For Both Generate And Training Process ------------------------------------------- #
    def set_model(self) -> None:        
        # Replace specific arguments with default model name
        select_default_indicators = ['*', 'default', 'default.h5', '*.*', '__pycache__']
        if self.arguments['model'] in select_default_indicators:
            self.arguments['model'] = c.DEFAULT_MODEL_FILENAME
        
        # Create name properly to avoid missing formats and put all to one dict
        name_split = os.path.splitext(self.arguments['model'])
        name_formatting = {}
        name_formatting['name'] = name_split[0] if name_split[0] else c.DEFAULT_MODEL_NAME
        name_formatting['extension'] = name_split[1] if name_split[1] else c.DEFAULT_MODEL_FORMAT
        name_formatting['filename'] = name_formatting['name'] + name_formatting['extension']
        # Replace self arguments with new name formatting
        self.arguments['model'] = name_formatting
        
        # Set models root from JSON
        models_root = self.json_search_objects(self.data_folder_json, ['name'], 'models')

        # Find every first child objects from JSON models object
        models_found = []
        for data in models_root:
            for model in data['children']:
                if model['type'] != 'json' and model['name'] != c.PATH_CHECKPOINTS_FOLDER:
                    models_found.append(model['name'])
                    
        # List all found models
        print(f"\nMODELS FOUND: {len(models_found)}")
        print(*models_found)
        
        # Report active model and overwrite in training modes
        if self.arguments['generate']:
            
            # Select argument provided model
            if self.arguments['model']['filename'] in models_found:
                print('Active Model:'.upper(), self.arguments['model']['filename'])
            
            # Select Default model
            elif c.DEFAULT_MODEL_FILENAME in models_found:
                self.arguments['model']['filename'] = c.DEFAULT_MODEL_FILENAME
                self.arguments['model']['name'] = c.DEFAULT_MODEL_NAME
                self.arguments['model']['extension'] = c.DEFAULT_MODEL_FORMAT
                
                print('Active Model:'.upper(), self.arguments['model']['filename'])     
                
            # Select first found model if paremeter and default model missing
            elif len(models_found) > 0:
                found_model_name = os.path.splitext(models_found[0])
                
                if len(found_model_name):
                    self.arguments['model']['filename'] = models_found[0]
                    
                if len(found_model_name) > 1:
                    self.arguments['model']['name'] = found_model_name[0]
                    self.arguments['model']['extension'] = found_model_name[1].replace('.', '')
                
                print(f"WARNING: Model name parameter missing. Selected the first model found in the '{c.PATH_MODELS_FOLDER}' folder. To select a specific model, use the '-c model_name' command line argument.")
                print('Active Model:'.upper(), self.arguments['model']['filename'])
        else:
            
            if self.arguments['model']['filename'] in models_found:
                print(f"WARNING: Model with name '{self.arguments['model']['filename']}' already exist. Trying to continue from that.")
                self.model_found = True      
                          
            else:
                print(f"New model '{self.arguments['model']['filename']}' will be saved to folder '{c.PATH_MODELS_FOLDER}'.")
                                
    
    # Save Model File ---------------------------------------------------------------------------- #
    def save_model_file(self, filename=None, save_json=True):
        filename = filename if filename != None else self.arguments['model']['filename']
        model_file_path = os.path.join(self.data_folder_path, c.PATH_MODELS_FOLDER, filename)
        
        # Save in chosen format (Default format: .h5)
        self.model.save(model_file_path)
        
        # Save model arguments to JSON file. Otherwise in generation process unique character count leads error if not same as in traning data.
        # So with this we make sure that we use same text files set as in training data.
        if save_json:
            json_model_object = json.dumps(self.arguments, indent=4)
            with open(model_file_path + '.json', "w") as json_model_info:
                json_model_info.write(json_model_object)
    
    
    # Load Model File ---------------------------------------------------------------------------- #
    def load_model_file(self):        
        # Get model name and load model from file
        model_file_path = os.path.join(self.data_folder_path, c.PATH_MODELS_FOLDER, self.arguments['model']['filename'])
        self.model = keras.models.load_model(model_file_path)
    
        # Load model arguments. Otherwise in generation process unique character count leads error if not same as in traning data.
        json_model_object = {}
        with open(model_file_path + '.json', "r") as json_model_info:
            json_model_object = json.load(json_model_info)
            
        # Copy values from info JSON
        self.arguments['items'] = json_model_object['items']
        self.arguments['sequence_length'] = json_model_object['sequence_length']
    
        # Print out all arguments
        print('\nProcess Arguments:'.upper())
        for k, v in self.arguments.items():
            print(k + ':', v)
    
    
    # Update Existing Dictionary With New Values ------------------------------------------------- #
    def update_dict(self, existing_dict, new_dict):
        for key, value in new_dict.items():
            if key in existing_dict:
                existing_dict[key] = value
        return existing_dict
    
    
    # Set Options From Terminal Command ---------------------------------------------------------- #
    def set_options(self) -> None:
        args_list = self.arguments['options']
        args_dict = {}
        for arg in args_list:
            key, value = arg.split('=')
            args_dict[key] = value
        self.arguments['options'] = self.update_dict(self.arguments, args_dict)
    
    
    # ============================================================================================ #
    # UTILITIES                                                                                    #
    # ============================================================================================ #
    
    # Get Path To Data Folder -------------------------------------------------------------------- #
    def get_data_folder_path(self):
        
        data_folder_path = os.path.abspath(os.path.join(Path.cwd(), c.PATH_DATA_FOLDER))
        if os.path.exists(data_folder_path):
            return data_folder_path
        else:
            print(c.PATH_DATA_FOLDER, 'folder not exist in location:', data_folder_path)     
            return None
    
    
    # Get/Create JSON File From Data Folder ------------------------------------------------------ #
    def cra_data_folder_json(self, path='', save_json=True):
        path = self.data_folder_path if not os.path.exists(path) else path        
        json_data = {'name': os.path.basename(path)}
        
        # https://stackoverflow.com/a/25226267/1629596
        if os.path.isdir(path):
            json_data['type'] = 'directory'
            json_data['path'] = path + '/'
            # Get extra info from JSON file (info.json) in author folders for e.g. genre
            if os.path.isfile(path + '/' + c.JSON_INFO_FILENAME):
                with open(path + '/' + c.JSON_INFO_FILENAME) as json_info_file:
                    author_data_json = json.load(json_info_file)
                    for i in author_data_json:
                        json_data[i] = author_data_json[i]                    
            # Go recursive childrens
            json_data['children'] = [self.cra_data_folder_json(os.path.join(path,x)) for x in os.listdir(path)]
        else:
            if os.path.splitext(path)[1] == '.json':
                json_data['type'] = 'json'
            elif os.path.splitext(path)[1] == '.txt':
                json_data['type'] = 'txt'
            else:
                json_data['type'] = 'file'
        
        # Serializing json
        json_object = json.dumps(json_data, indent=4)
            
        # Writing to sample.json if needed
        if save_json:
            with open(os.path.join(c.PATH_DATA_FOLDER, 'data_folder.json'), 'w') as json_file:
                json_file.write(json_object)

        return json_data    
    
    
    # Search From Multidimensional JSON Data ----------------------------------------------------- #
    def json_search_objects(self, obj, key=None, *args):
        # Conver args to lowercase to prevent user typos
        args_lowercase = self.flatten_list(args[0], True, True)
        
        # Select all author -t indicator used if some of these arguments found
        select_all_indicators = ['*', 'all', '*.*', '__pycache__', ' ', None]
        if args_lowercase[0] in select_all_indicators:
            # print(f'INFO: Train argument with select all indicator ({args_lowercase}) detected.')
            args_lowercase = ['*']
        
        key = [] if key is None else key
        lst = []
        
        def extract(obj, key, lst):        
            # Recursively search for values of key in JSON tree
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in key:
                        # Check if value is list type e.g. multiple genres
                        if isinstance(v, list):
                            for i in v:
                                if i.lower() in args_lowercase:
                                    lst.append(obj)
                        # Search author name with lowercase conversion
                        elif v.lower() in args_lowercase:
                            lst.append(obj)
                    elif isinstance(v, (dict, list)):
                        extract(v, lst, key)
            elif isinstance(obj, list):
                for item in obj:
                    extract(item, lst, key)
            return lst    
        return extract(obj, key, lst)
    
        
    # Clean String ------------------------------------------------------------------------------- #
    def clean_string(self, string):        
        string = string.lower()
        cleaned_text = []
        
        for line in string.splitlines():
            cleaned_line = str(line)
            # Remove empty lines            
            if cleaned_line.strip() and not cleaned_line.isspace():
                # Keep only characters that is in allowed list
                cleaned_line = ''.join([s for s in cleaned_line if s in self.allowed_characters])
                # Replace some marks that is similar to each other to reduce characters
                cleaned_line = self.replace_characters_with_character(cleaned_line, '‘’´`', "'")
                cleaned_line = self.replace_characters_with_character(cleaned_line, '“”', '"')
                cleaned_line = self.replace_characters_with_character(cleaned_line, '–—•▪•◆', '-')
                # Remove square bracket words
                cleaned_line = self.remove_words_within_square_brackets(cleaned_line)
                # Remove multiple whitespaces
                cleaned_line = ' '.join(cleaned_line.split())
                # Add one dot and later remove all consecutive dots
                cleaned_line += '.'
                cleaned_line = self.replace_consecutive_characters(cleaned_line, '.')                
                cleaned_text.append(cleaned_line)
                
        # Remove any empty strings from the list
        cleaned_text = list(filter(lambda x: x.strip() and x.strip('.') != '', cleaned_text))
        return ' '.join(cleaned_text)
    
    
    # Replace All Characters Found On List With Chosen Character --------------------------------- #
    def replace_characters_with_character(self, string, remove_chars, replacement_char):
        new_string = ''
        remove_chars = list(remove_chars)
        for char in string:
            if char in remove_chars:
                new_string += replacement_char
            else:
                new_string += char        
        return new_string
    
    
    # Remove Words And Brackets That Are Inside Of Square Brackets ------------------------------- #
    def remove_words_within_square_brackets(self, text):
        words = text.split()
        result = ''
        in_brackets = False

        for word in words:
            if '[' in word:
                in_brackets = True
                word = ''
            if ']' in word:
                in_brackets = False
                word = ''
            if not in_brackets:
                result += word + ' '
        
        return result.strip()
    

    # Format Text With Adding New Line Everytime With Period ------------------------------------- #
    def format_text(self, string):
        splitted_text = string.split('.')
        disallowed_initial_characters = " '\"’,.?!-"
        formatted_text = ''
        for s in splitted_text:
            # Remove disallowed initial characters
            if len(s) > 0:
                if s[0] in disallowed_initial_characters:
                    s = self.delete_char(s, [0])
            # Remove whispaces, capitalize and add dot and new line to end.
            formatted_text += s.strip().capitalize()
            if len(s) > 0:
                formatted_text += ".\n"
        return formatted_text


    # Replace Consecutive Characters ------------------------------------------------------------- #
    def replace_consecutive_characters(self, s, target_char):
        if len(s) < 2:
            return s
        new_str = [s[0]]
        for char in s[1:]:
            if char == target_char and new_str[-1] == target_char:
                continue
            new_str.append(char)
        return ''.join(new_str)
    
    
    # Remove Repeated_characters ----------------------------------------------------------------- #
    def remove_repeated_characters(self, s, ch, target_char):
        new_str = []
        prev_char = None
        for char in s:
            if char == target_char:
                if char == ch and prev_char == ch:
                    continue
                else:
                    new_str.append(char)
                    prev_char = char
        return ''.join(new_str)
    
    
    # Deletes All The Indexes From The String And Returns The New One ---------------------------- #
    def delete_char(self, string, indexes):
        return ''.join((char for idx, char in enumerate(string) if idx not in indexes))


    # Flatten Multidimensional List -------------------------------------------------------------- #
    def flatten_list(self, l, lowercase=False, remove_duplicates=False):
        # Flattens multidimensional list to 1D list.
        # https://stackoverflow.com/a/73304966/1629596
        output = []
        def flatten_recursive(v):
            if isinstance(v, str):
                # Remove extra whitespaces from string
                if lowercase: v = v.lower()
                output.append(' '.join(v.strip().split()))
            if isinstance(v, (int, bool)):
                output.append(v)
            if isinstance(v, (dict, list, tuple)):
                for i in range(0, len(v)):
                    flatten_recursive(v[i])
        flatten_recursive(l)
        # Remove duplicates from list if needed
        if remove_duplicates: output = list(dict.fromkeys(output))
        return output
    
    
    # Save Different Type of Text Files -------------------------------------------------------------- #
    def save_text_file(self, data, path=Path.cwd(), name=None, extension='txt', prefix='', suffix='', merge_results=False):                
        merge_results = 'a' if merge_results else 'w'
        file_path = ''
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"WARNING: '{path}' not exist. Created new folder.")
        
        if not name:
            timestamp = datetime.datetime.now().strftime('%m%d%Y%H%M%S%f')
            file_path = os.path.join(path, str(timestamp))
        else:
            file_path = os.path.join(path, str(name))
            
        file_path = prefix + file_path + suffix
                          
        # Save numpy array in text format
        if isinstance(data, np.ndarray):
            # https://stackoverflow.com/a/18145279/1629596
            data.tofile(file_path, sep=" ", format="%s")
        # Save normal file
        else:
            with open(file_path + '.' + extension, merge_results) as data_file:
                data_file.write("\n"+str(data))
            

# ================================================================================================ #
# MAIN                                                                                             #
# ================================================================================================ #
    
def main():
    
    arguments = {
        # Generate Arguments
        'generate': False,
        'primer': '',
        'text_formatting': c.TEXT_FORMATTING,
        'temperature': c.TEMPERATURE,
        'texts_length': c.GENERATE_TEXT_LENGTH,
        'texts_count': c.GENERATE_TEXTS_COUNT,
        'merge_results': c.MERGE_RESULTS,
        # Train Argumantes
        'train': False,
        'evaluate': False,
        'model': c.DEFAULT_MODEL_FILENAME,
        'items': ['all'],        
        'sequence_length': c.SEQUENCE_LENGTH,
        'checkpoints': c.USE_CHECKPOINTS,
        'step_size': c.STEP_SIZE,
        'batch_size': c.BATCH_SIZE,
        'epochs': c.EPOCHS,
        'learning_rate': c.LEARNING_RATE,
        'lstm_units': c.LSTM_UNITS,
        'validation': c.USE_VALIDATION,
        'validation_split': c.VALIDATION_SPLIT,
        'shuffle_data': c.SHUFFLE_DATA,
        'steps_per_epoch': c.STEPS_PER_EPOCH,
        'loss_function': c.LOSS_FUNCTION,
        'monitor_metric': c.MONITOR_METRIC,
        'train_patience': c.TRAIN_PATIENCE,
        'activation_layer': c.ACTIVATION_LAYER,
        'debug': c.USE_DEBUG,
        'options': [],
    }
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'gtep:l:t:c:i:m:o:v', ['generate', 'train', 'evaluate', 'primer=', 'texts_length=', 'texts_count=', 'items=', 'model=', 'options=', 'version'])
        
    except getopt.GetoptError as e:
        print('ERROR: ', str(e))
        sys.exit(2)

    for opt, arg in opts:
        
        if opt in ('-g', '--generate'):
            arguments['generate'] = True
               
        if opt in ('-t', '--train'):
            arguments['train'] = True
            
        if opt in ('-e', '--evaluate'):
            arguments['evaluate'] = True
            
        if opt in ('-p', '--primer'):
            arguments['primer'] = arg if arg else c.PRIMER_TEXT
            
        if opt in ('-l', '--length'):
            arguments['texts_length'] = arg if arg else c.GENERATE_TEXT_LENGTH
            
        if opt in ('-c', '--count'):
            arguments['texts_count'] = arg if arg else c.GENERATE_TEXTS_COUNT
            
        if opt in ('-i', '--items'):
            arguments['items'] = list(map(str.strip, arg.split(',')))
            
        if opt in ('-m', '--model'):
            arguments['model'] = arg if arg else c.DEFAULT_MODEL_FILENAME
            
        if opt in ('-o', '--options'):
            arguments['options'] = list(map(str.strip, arg.split(',')))
            
        if opt in ('-v', '--version'):
            print('Python:', sys.version)
            print('Numpy:', np.__version__)
            print('Keras:', keras.__version__)
            exit()

    TextGenerator(arguments)
    
if __name__ == '__main__':
    main()