#!/usr/bin/env python

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START speech_transcribe_infinite_streaming]
from __future__ import division

import time
import re
import sys

from google.cloud import speech_v1 as speech

import pyaudio
from six.moves import queue

import os
credential_path = "C:\\Users\Mamadou\Documents\Cours\M1 ATAL\S2\TER\Google Speech API\speechtotextapi-340414-1e2134d625a1.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

# imports pour le WordCloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# imports pour la traduction (requetes http envoyées au service de traduction, ici Google Traduction mais possible d'adapter pour d'autres services, eg. DeepL ou FreeTranslate)
import requests
import json

# Les différentes clés ici
#import config


#in_language = "en-US"
in_language = "fr-FR"
languages = ["fr","en","es","de"]

nb_phrases_memoire = 10

add_in_pos = 0
texte_complet=dict()
for l in languages:
    texte_complet[l] = ["" for i in range(nb_phrases_memoire)]

#  mots à exclure (ici en français uniquement, probablement que ça existe ailleurs et pour les autres langues, à voir)
exclure_mots = dict()
for l in languages:
    exclure_mots[l] = []
exclure_mots["fr"] = ['d', 'du', 'de', 'la', 'des', 'le', 'et', 'est', 'elle', 'une', 'en', 'que', 'aux', 'qui', 'ces', 'les', 'dans', 'sur', 'l', 'un', 'pour', 'par', 'il', 'ou', 'à', 'ce', 'a', 'sont', 'cas', 'plus', 'leur', 'se', 's', 'vous', 'au', 'c', 'aussi', 'toutes', 'autre', 'comme']


# un masque  joli pour le nuage
"""
mask = np.array(Image.open("nuage_mask.png"))
mask[mask == 1] = 255
"""

# Audio recording parameters
STREAMING_LIMIT = 55000
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms

"""
def translate(txt,lfrom,lto):
    lfrom = lfrom[0:2]
    lto = lto[0:2]
    if lfrom == lto:
        return txt
    txt=txt.replace("'","’")
    txt=txt.replace('"',"’’")
    proxyDict = {"http"  : config.PROXYHTTP, "https" : config.PROXYHTTPS}
    url = "https://translation.googleapis.com/language/translate/v2?key="+config.GOOGLEAPITOKEN
    datapost={'q':txt, 'source':lfrom, 'target':lto, 'format':'text'}
    if config.PROXYHTTP != "":
        result = requests.post(url,data=datapost,proxies=proxyDict)
    else:
        result = requests.post(url,data=datapost)
    result=json.loads(result.text)
    if (not result.get('data')) or (not result.get('data').get('translations')[0]) or (not result.get('data').get('translations')[0].get('translatedText')):
        return ""
    else:
        return result.get('data').get('translations')[0].get('translatedText')



def draw_wc(text):
    global in_language,languages,texte_complet, mask, add_in_pos, languages,exclure_mots
    for l in languages:
        texte_complet[l][add_in_pos % len(texte_complet)] = translate(text,in_language,l)
        wordcloud = WordCloud(font_path="Pacifico.ttf", background_color = 'white', min_word_length=4, width=800, height=600, contour_width= 5, stopwords = exclure_mots[l], max_words = 50, mask = mask).generate(" ".join(texte_complet[l]))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.savefig("output_"+l+".png")
    add_in_pos += 1

"""

def get_current_time():
    return int(round(time.time() * 1000))


def duration_to_secs(duration):
    return duration.seconds + (duration.nanos / float(1e9))

# Ceci vient de l'exemple fourni par google.

class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk_size):
        self._rate = rate
        self._chunk_size = chunk_size
        self._num_channels = 1
        self._max_replay_secs = 5

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()

        # 2 bytes in 16 bit samples
        self._bytes_per_sample = 2 * self._num_channels
        self._bytes_per_second = self._rate * self._bytes_per_sample

        self._bytes_per_chunk = (self._chunk_size * self._bytes_per_sample)
        self._chunks_per_second = (
                self._bytes_per_second // self._bytes_per_chunk)

    def __enter__(self):
        self.closed = False

        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, *args, **kwargs):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            if get_current_time() - self.start_time > STREAMING_LIMIT:
                self.start_time = get_current_time()
                break
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)


def listen_print_loop(responses, stream):
    responses = (r for r in responses if (
            r.results and r.results[0].alternatives))

    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.

        sentence = ""
        if response.results[0].stability >= 0.70:
            sentence += " "+response.results[0].alternatives[0].transcript

        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        top_alternative = result.alternatives[0]
        transcript = top_alternative.transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
            #sys.stdout.write(transcript + overwrite_chars + '\r')
            print("héhé")
            sys.stdout.write(sentence + "blablabla "+overwrite_chars + '\r')
            sys.stdout.flush()

            #num_chars_printed = len(transcript)
            num_chars_printed = len(sentence)
        else:
#        if result.is_final:
            #print(transcript + overwrite_chars)
            print("haha")
            print(sentence + overwrite_chars)
            #draw_wc(transcript + overwrite_chars) # modification de l'exemple ici
            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r'\b(fin du cours|fin de la transcription)\b', transcript, re.I):
                print('Transcription terminée..')
                stream.closed = True
                break

            num_chars_printed = 0


def main():
    global in_language
    #os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./key.json"
    client = speech.SpeechClient()
    config = speech.types.RecognitionConfig(
        #encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        encoding=speech.types.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code=in_language,
        max_alternatives=3,
        enable_word_time_offsets=True)
    streaming_config = speech.types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)

    print('Transcription en '+config.language_code)
    print('Dites "fin du cours" pour sortir.')

    with mic_manager as stream:
        while not stream.closed:
            audio_generator = stream.generator()
            requests = (speech.types.StreamingRecognizeRequest(
                audio_content=content)
                for content in audio_generator)

            responses = client.streaming_recognize(streaming_config,
                                                   requests)
            # Now, put the transcription responses to use.
            listen_print_loop(responses, stream)


if __name__ == '__main__':
    main()
# [END speech_transcribe_infinite_streaming]
