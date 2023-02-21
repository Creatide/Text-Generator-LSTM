'''
ABOUT THIS SCRIPT:
------------------
This script is not related to actual code, but is an additional script I use to 
detect the languages in text files that I use for training. Its purpose is to 
identify the languages used in the files and help detect any incorrect languages. 
This script is also helpful in cleaning up files that could potentially interfere 
with the training data for the target language.

INSTALL PROCESS:
----------------
Update all of your packages in Linux first:
sudo apt update
sudo apt upgrade
 
To use pycld2 you need to install gcc and g++ first:
sudo apt-get install gcc
sudo apt-get install g++
 
Install regex:
pip install regex
 
Install pycld2:
python -m pip install -U pycld2
'''
 
import os, fnmatch
import pycld2 as cld2
import datetime
import regex
 
# Folder that will be scanned for files
scan_folder = '.././data/texts'
# What type of text files looking forcd
search_file_type = 'txt'
# Search these languages at least
search_languages = ['English']
# Skip languages that found this list
# Also can be use for invert search, so every file that not contains this language
skip_languages = ['Scots']
# Filter minimum count for the number of languages that need to be in file
min_lang_count = 2
# CAUTION! Use with care, because this will delete files!
delete_result_files = False
 
 
files_info = []
search_languages = [item.lower() for item in search_languages]
skip_languages = [item.lower() for item in skip_languages]
RE_BAD_CHARS = regex.compile(r"[\p{Cc}\p{Cs}]+")
 
# https://stackoverflow.com/a/13299851/1629596
def find_files (path, filter):
    for root, dirs, files in os.walk(path):
        for file in fnmatch.filter(files, filter):
            yield os.path.join(root, file)
            
def save_file_info(data=None, name=datetime.datetime.now().strftime('%m%d%Y%H%M%S'), extension='txt', prefix='', suffix='', remove_files=delete_result_files):
    if not data:
        print('No data provided for saving process.')
        exit()
    name = prefix + name + suffix + '.' + extension
    
    # Sort items: https://stackoverflow.com/a/73050/1629596
    data = sorted(data, key=lambda d: d['file'].name) 
    
    for item in data:        
        result_file = []
        
        # Search languages from details data
        found_languages = []
        for lang in item['details']:
            if lang[0] != 'Unknown':
                found_languages.append(lang[0].lower())
        
        if min_lang_count <= len(found_languages):
            if len(search_languages) and not any(x in skip_languages for x in found_languages):
                if any(x in search_languages for x in found_languages):
                    result_file.append({'filename':  item['file'].name, 'languages': found_languages})
            elif not any(x in skip_languages for x in found_languages):
                result_file.append({'filename':  item['file'].name, 'languages': found_languages})
 
        if remove_files and len(result_file):
            print('File deleted:', str(result_file[0]['filename']))
            os.remove(str(result_file[0]['filename']))
            
        # Write new lines to file
        if len(result_file):
            with open(name, 'a') as f:
                write_line = str(result_file[0]['filename']) + '\n' + str(result_file[0]['languages']) + '\n'
                f.write(write_line)
                
 
# https://github.com/aboSamoor/polyglot/issues/71#issuecomment-707997790
def remove_bad_chars(text):
    return RE_BAD_CHARS.sub("", text)
    
for single_file in find_files(scan_folder, '*.' + search_file_type):  
    with open(single_file, 'r') as file:
        text = remove_bad_chars(file.read())
        is_reliable, text_bytes, details = cld2.detect(text, isPlainText=True)
        file_info = {'file': file, 'details': list(details)}
        files_info.append(file_info)
 
save_file_info(files_info, 'Files_Languages_Overview')
 
 