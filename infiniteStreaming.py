#!/usr/bin/env python

# Copyright 2019 Google LLC
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

"""Google Cloud Speech API sample application using the streaming API.
NOTE: This module requires the dependencies `pyaudio` and `termcolor`.
To install using pip:
    pip install pyaudio
    pip install termcolor
Example usage:
    python transcribe_streaming_infinite.py
"""
from socket import timeout
from google.cloud import translate
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from random import randrange

# [START speech_transcribe_infinite_streaming]

import re
import sys
import time
import os

from google.cloud import speech
import pyaudio
from six.moves import queue

credential_path = "C:\\Users\Mamadou\Documents\Cours\M1 ATAL\S2\TER\Google Speech API\speechtotextapi-340414-1e2134d625a1.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

# Audio recording parameters
STREAMING_LIMIT = 55000
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"


def get_current_time():
    """Return Current Time in MS."""

    return int(round(time.time() * 1000))


class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk_size):
        self._rate = rate
        self.chunk_size = chunk_size
        self._num_channels = 1
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

    def __enter__(self):

        self.closed = False
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
        """Stream Audio from microphone to API and to local buffer"""

        while not self.closed:
            data = []

            if self.new_stream and self.last_audio_input:

                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)

                if chunk_time != 0:

                    if self.bridging_offset < 0:
                        self.bridging_offset = 0

                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time

                    chunks_from_ms = round(
                        (self.final_request_end_time - self.bridging_offset)
                        / chunk_time
                    )

                    self.bridging_offset = round(
                        (len(self.last_audio_input) - chunks_from_ms) * chunk_time
                    )

                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])

                self.new_stream = False

            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.

            #comment this section so the api won't auto stop when bad connexion or nothing is provided 
            
            chunk = self._buff.get()
            self.audio_input.append(chunk)

            if chunk is None:
                return
            
            data.append(chunk)
            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    
                    if chunk is None:
                        return
                    
                    data.append(chunk)
                    self.audio_input.append(chunk)

                except queue.Empty:
                    break

            yield b"".join(data)


def listen_print_loop(responses, stream):
    """Iterates through server responses and prints them.
    The responses passed is a generator that will block until a response
    is provided by the server.
    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.
    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """

    for response in responses:
        """
        if get_current_time() - stream.start_time > STREAMING_LIMIT:
            stream.start_time = get_current_time()
            break
        """
        sentence = ""
        if not response.results:
            continue
        else:
            """
            print("------------------QUE CACHE RESPONSE ---------------------")
            print(response)
            #sentence = response.
            print("----------------- FIN RESPONSE --------------------------")
            print("================= PRINTING RESPONSE RESULTS ===============")
            print(response.results[0])
            """
        if response.results[0].stability >= 0.70:
            sentence += " "+response.results[0].alternatives[0].transcript
        
        #print("SENTENCE:", sentence)
        #print("+++++++++++++++++++++++++++++++")
        result = response.results[0]
        
        
        if not result.alternatives:
            continue
        
        transcript = result.alternatives[0].transcript

        result_seconds = 0
        result_micros = 0

        if result.result_end_time.seconds:
            result_seconds = result.result_end_time.seconds

        if result.result_end_time.microseconds:
            result_micros = result.result_end_time.microseconds

        stream.result_end_time = int((result_seconds * 1000) + (result_micros / 1000))

        corrected_time = (
            stream.result_end_time
            - stream.bridging_offset
            + (STREAMING_LIMIT * stream.restart_counter)
        )
        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.

        if result.is_final:

            sys.stdout.write(GREEN)
            sys.stdout.write("\033[K")
            sys.stdout.write(str(corrected_time) + ": " + transcript + "\n")
            #sys.stdout.write(str(corrected_time) + ": " + sentence + "\n")

            stream.is_final_end_time = stream.result_end_time
            stream.last_transcript_was_final = True

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            #if re.search(r"\b(sortir|quit)\b", transcript, re.I):
            if re.search(r"\b(sortir|quit)\b", transcript, re.I):
                sys.stdout.write(YELLOW)
                sys.stdout.write("Exiting...\n")
                stream.closed = True
                break

        else:
            sys.stdout.write(RED)
            sys.stdout.write("\033[K")
            #sys.stdout.write(str(corrected_time) + ": " + transcript + "\r")
            sys.stdout.write(str(corrected_time) + ": " + "sentence" + "\r")

            #added
            fhand = open("streaming.txt", "a+")
            #for translation in response.translations:
            fhand.write("{}".format(transcript))
            #end added

            stream.last_transcript_was_final = False


def main():
    """start bidirectional streaming from microphone input to speech API"""

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="en-US",
        max_alternatives=1,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)
    print(mic_manager.chunk_size)
    sys.stdout.write(YELLOW)
    sys.stdout.write('\nListening, say "Quit" or "Exit" to stop.\n\n')
    sys.stdout.write("End (ms)       Transcript Results/Status\n")
    sys.stdout.write("=====================================================\n")

    with mic_manager as stream:

        while not stream.closed:
            sys.stdout.write(YELLOW)
            sys.stdout.write(
                "\n" + str(STREAMING_LIMIT * stream.restart_counter) + ": NEW REQUEST\n"
            )

            stream.audio_input = []
            audio_generator = stream.generator()

            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )

            responses = client.streaming_recognize(streaming_config, requests, timeout=72000000)

            # Now, put the transcription responses to use.
            listen_print_loop(responses, stream)

            if stream.result_end_time > 0:
                stream.final_request_end_time = stream.is_final_end_time
            stream.result_end_time = 0
            stream.last_audio_input = []
            stream.last_audio_input = stream.audio_input
            stream.audio_input = []
            stream.restart_counter = stream.restart_counter + 1

            if not stream.last_transcript_was_final:
                sys.stdout.write("\n")
            stream.new_stream = True


english_stop_words = [
    'a','about','above','after','again','against','all','also','am','an',
    'and','any','are','aren\'t','as','at','be','because','been','before',
    'being','below','between','both','but','by','can','can\'t','cannot',
    'com','could','couldn\'t','did','didn\'t','do','does','doesn\'t',
    'doing','don\'t','down','during','each','else','ever','few','for',
    'from','further','get','had','hadn\'t','has','hasn\'t','have',
    'haven\'t','having','he','he\'d','he\'ll','he\'s','hence','her',
    'here','here\'s','hers','herself','him','himself','his','how','how\'s',
    'however','http','i','i\'d','i\'ll','i\'m','i\'ve','if','in','into',
    'is','isn\'t','it','it\'s','its','itself','just','k','let\'s','like',
    'me','more','most','mustn\'t','my','myself','no','nor','not','of',
    'off','on','once','only','or','other','otherwise','ought','our',
    'ours','ourselves','out','over','own','r','same','shall','shan\'t',
    'she','she\'d','she\'ll','she\'s','should','shouldn\'t','since',
    'so','some','such','than','that','that\'s','the','their','theirs',
    'them','themselves','then','there','there\'s','therefore','these',
    'they','they\'d','they\'ll','they\'re','they\'ve','this','those',
    'though','through','to','too','under','until','up','very','was',
    'wasn\'t','we','we\'d','we\'ll','we\'re','we\'ve','were','weren\'t',
    'what','what\'s','when','when\'s','where','where\'s','which','while',
    'who','who\'s','whom','why','why\'s','with','won\'t','would',
    'wouldn\'t','www','you','you\'d','you\'ll','you\'re','you\'ve',
    'your','yours','yourself','yourselves'
]
french_stop_words = [
                'd\'','du','de', 'la', 'des', 
                'le', 'et', 'est', 'elle', 'une', 'en', 'que', 
                'aux', 'qui', 'ces', 'les', 'dans', 'sur', 'l\'', 
                'un', 'pour', 'par', 'il', 'ou', 'à', 'ce', 'a', 
                'sont', 'cas', 'plus', 'leur', 'se', 's', 'vous', 
                'au', 'c', 'aussi', 'toutes', 'autre', 'comme', 'd\'un',
                'nos', 'vos', 'leur', 'mon', 'ton', 'son', 'ma', 'ta', 'sa',
                'mes', 'tes', 'ses', 'notre', 'votre','l\'huile', 'l\'eau',
                'c\'est', '1', '2', '3','4','5','6','7','8','9','0', 'c', 'c\''
            ]

def frequency(file, langage_stop_words):
    frequency = {}
    fhand = open(file, "r")
    """
    # handle the file if it is a long blob
    for line in fhand:
        sp_line = line.split()
        sp_line = list(set(sp_line).difference(langage_stop_words))
        for word in sp_line:
            frequency[word] = frequency.get(word, 0) + 1
    """
    text = fhand.read() # read the whole file
    sp_text = text.split()
    print(sp_text)
    sp_text = list(set(sp_text).difference(langage_stop_words))
    for word in sp_text:
        frequency[word] = frequency.get(word, 0) + 1
    frequency_to_string = " ".join(sp_text)
    print("=======================")
    #print(frequency_to_string)
    
    print(frequency)
    fhand = open(file, "w")
    fhand.write(frequency_to_string)
    return frequency

def translate_text(text, project_id="speechtotextapi-340414"):
    """Translating Text."""

    client = translate.TranslationServiceClient()

    location = "global"

    parent = f"projects/{project_id}/locations/{location}"

    # Translate text from English to French
    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": "en-US",
            "target_language_code": "fr-FR",
        }
    )

    # Display the translation for each input text provided
    # Write transcription in a file 
    fhand = open("translateStreaming.txt", "w")
    for translation in response.translations:
        fhand.write("{}".format(translation.translated_text))
        print("Translated text: {}".format(translation.translated_text))
    #time.sleep(0.5)


def makeImage(text):
    wc = WordCloud(background_color="white")
    wc.generate_from_frequencies(text)
    
    plt.imshow(wc, interpolation = "bilinear")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
    """
    fhand = open("streaming.txt", "r")
    text = fhand.read()
    print("Start of the translation")
    translate_text(text) # write in translateStreaming.txt
    file = os.path.abspath("translateStreaming.txt")
    freq = frequency(file, english_stop_words)

    fhand = open("translateStreaming.txt", "r")
    words = fhand.read()
    wordcloud = WordCloud(background_color = 'white', stopwords = french_stop_words).generate(words)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

    makeImage(freq)
    """

