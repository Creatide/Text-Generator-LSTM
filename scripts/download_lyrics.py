'''
# ================================================================================================ #
# LYRICGENIUS SCRIPT: DOWNLOAD AND CONVERT TO TEXT FILES                                           #
# ================================================================================================ #

This script downloads song lyrics using the 'LyricsGenius' python library and converts them into text 
files by removing unnecessary special characters. The purpose of the script is to prepare the lyrics 
as better data to be used for training.

To use this script you need to install LyricGenius and language detection library first.
 
https://lyricsgenius.readthedocs.io/en/master/usage.html
 
# INSTALL LIBRARIES: ----------------------------------------------------------------------------- #

LyricGenius:
pip install lyricsgenius
 
Language detection tool:
pip install langdetect 

# GENIUS TOKEN ----------------------------------------------------------------------------------- #

First you'll need to sign up for a (free) account that authorizes access to the Genius API. After 
signing up/ logging in to your account, head out to the API section on Genius and create a new API 
client. After creating your client, you can generate an access token to use with the library. 

https://genius.com/api-clients

'''
 
import os
import json
from lyricsgenius import Genius
from langdetect import detect
from langdetect import DetectorFactory
from difflib import SequenceMatcher
DetectorFactory.seed = 0
 
class Lyrics():
    def __init__(self):
        
        # Artist Name And ID --------------------------------------------------------------------- #
        self.artist_name = 'Nirvana'
        # Use also artist ID if result is wrong artist
        # Get artist ID from genius artist page source
        # Use None by default
        self.artist_id = None
        
        # Settings ------------------------------------------------------------------------------- #
        self.save_only_language = 'en'
        # Prevent shortest lyrics to be saved
        self.min_lyric_length = 200
        # Prevent duplicates with similarity check
        # If name is more than 0.3 (30%) similar, skip
        self.similarity_ratio = 0.7
        
        # Initialize Filename For JSON ----------------------------------------------------------- #
        self.json_filename = self.clean_name('Lyrics_' + self.clean_artist_name(self.artist_name) + '.json')
        self.json_filename = self.json_filename.replace(' ', '')
        
        # DISABLE DOWNLOADING AND ONLY CLEAN TEXT FILES ------------------------------------------ #
        self.download_lyrics()
        
        # Remove These Characters From Lyrics ---------------------------------------------------- #
        self.remove_symbols = {
            'д': 'ä', 'Д': 'Ä', 
            'ц': 'ö', 'Ц': 'Ö', 
            '/': '', '_': '', '--': '', '?': '',
            'I:': '', 'II:': '', 'III:': '',
            'San.': '', 'san.': '', 'Säv.': '', 'säv.': '',
            'Kertosäe2:': '', 'Kertosäe:': '' , 
            'Embed': '', 'You might also like': '', 
            'CHORUS': '', 'Chorus': '', '(chorus:)': '',
            'VERSE': '', 'Verse': '', 
            'Intro': '', 'INTRO': '', 
            'SOLOS:': '', 'SOLO:': '', '(solo)': '', '(Guitar solo)': '', 
            '(Missing Stanzas)': '',
            'PRE-:': '', 'PRE-': '',
            'C-PART': '', 'Spoken:': '',              
            '(x2)': '', '( x2)': '', '(2x)': '', '(x3)': '', '(3x)': '', '( x3)': '', 
            '()': '', '(?)': '', '(???)': '', 
            '(original)': '', 
            "Refren':": '', "Refrain:": '', "Refren'": '',
            '(NARRATION:)': '',
            'These lyrics have yet to be transcribed': '',
        }
        
        # Conver downloaded data to text files        
        self.convert_to_txt()
    
    # Download Lyrics ---------------------------------------------------------------------------- #    
    def download_lyrics(self, artist_id = None):
        if not os.path.exists(self.json_filename):
            genius = Genius('YOUR GENIUS TOKEN HERE')
            genius.remove_section_headers = True
            genius.response_format = 'plain'
            artist = genius.search_artist(artist_name=self.artist_name ,artist_id=self.artist_id, sort="title")
            if artist:
                artist.save_lyrics()
 
    # Extract Lyrics From JSON And Save To Separated Files --------------------------------------- #
    def convert_to_txt(self):
 
        # Check if json file exist
        if os.path.exists(self.json_filename):
 
            json_data = {}
            artist_data = []
 
            with open(self.json_filename) as json_file:
                json_data = json.load(json_file)
                print('### File Found:', self.json_filename)                
            
            # Get data from json for organizing
            for song in json_data['songs']:
                data = {
                    'artist': song['artist'],
                    'song': song['title'],
                    'lyrics': song['lyrics'],
                }
                artist_data.append(data)
                
            # Sort data list by song title
            artist_data = sorted(artist_data, key=lambda d: d['song'])
            
            # Remove similar/duplicate files from list
            artist_data_cleaned = []
            for artist in artist_data:
                duplicate_found = False                
                for item in artist_data_cleaned:
                    name_1 = self.remove_words_within_brackets(item['song'].lower())
                    name_2 = self.remove_words_within_brackets(artist['song'].lower())
                    if self.similar(name_1, name_2) > self.similarity_ratio:
                        print('### Duplicate song found:', name_1, 'is almost same as', name_2)
                        duplicate_found = True
                if duplicate_found == False:
                    artist_data_cleaned.append(artist)
            
            for song in artist_data_cleaned:
                self.save_text_file(song)
 
 
    # Save Text Files ---------------------------------------------------------------------------- #
    def save_text_file(self, data):
                                        
        filename = self.clean_name(data['song']) + ' - ' + self.clean_name(data['artist']) + '.txt'
        file_path = os.path.join(os.getcwd(), 'Lyrics', data['artist'])
        
        if not os.path.exists(file_path):
            os.makedirs(file_path)                
        file_path = os.path.join(file_path, filename)
        
        # Check if there is enough long lyrics and correct language
        if len(data['lyrics']) > self.min_lyric_length and detect(data['lyrics']) == self.save_only_language:
            with open(file_path, 'w', encoding='utf-8') as f:
                reformatted_data = ''
                for lyric in data['lyrics']:
                    reformatted_data += lyric
                # Try to split string from target word like "Lyrics"
                if len(reformatted_data.split('Lyrics', 1)) > 1:
                    reformatted_data = reformatted_data.split('Lyrics', 1)[1]
                # If needed replace character with another
                for k, v in self.remove_symbols.items():
                    reformatted_data = reformatted_data.replace(k, v)
                # Save file
                f.write(str(reformatted_data))
        else:
            print('### No proper lyrics for song:', data['song'])
        
    # Clean Names To Avoid Path Errors ----------------------------------------------------------- #
    def clean_name(self, name):        
        return_name = name
        remove_symbols = [*'<>:"/\|?*']
        for i in remove_symbols:
            return_name = return_name.replace(i, '')
        return return_name
    
    # Clean Artist Names To Make Path Work ------------------------------------------------------- #
    def clean_artist_name(self, name):        
        return_name = name
        remove_symbols = [*'<>:"/\|?*() ']
        for i in remove_symbols:
            return_name = return_name.replace(i, '')
        return return_name
    
    # Remove Words Within Brackets --------------------------------------------------------------- #
    def remove_words_within_brackets(self, text):
        words = text.split()
        result = ''
        in_brackets = False
        for word in words:
            if '(' in word:
                in_brackets = True
                word = ''
            if ')' in word:
                in_brackets = False
                word = ''
            if not in_brackets:
                result += word + ' '        
        return result.strip()
    
    # Compare Two Filenames And Return Similarity Value ------------------------------------------ #
    def similar(self, a, b):
        return SequenceMatcher(None, a, b).ratio()
    
    
# ================================================================================================ #
# MAIN                                                                                             #
# ================================================================================================ #
    
def main():
    Lyrics()
    
if __name__ == '__main__':
    main()