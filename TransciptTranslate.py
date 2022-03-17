from concurrent.futures import thread
import os
import io
from google.cloud import speech
from google.cloud import translate
from nltk.corpus import stopwords

import time
import threading

# WordCloud import
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from random import randrange

credential_path = "C:\\Users\Mamadou\Documents\Cours\M1 ATAL\S2\TER\Google Speech API\speechtotextapi-340414-1e2134d625a1.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

def streamFile(file):
    return os.path.abspath(file)


#audio_file = "OSR_us_000_0010_8k.wav"
audio_file = "OSR_fr_000_0041_8k.wav"
assembly = set()

def transcribe_streaming(stream_file):
    client = speech.SpeechClient()
    with io.open(stream_file, "rb") as audio_file:
        content = audio_file.read()
    
    # In practice, stream should be a generator yielding chunks of audio data.
    stream = [content]
    #print(stream)

    requests = (
        speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in stream
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=8000,
        language_code="fr-FR", #change en-US to fr-FR
    )

    streaming_config = speech.StreamingRecognitionConfig(config=config)

    # streaming_recognize returns a generator.
    responses = client.streaming_recognize(
        config=streaming_config,
        requests=requests,
    )
    fhand = open("transcript_file.txt", "w")
    print("================ RESPONSES ===============")
    for response in responses:
        #print(response)
        
        for result in response.results:
            alternatives = result.alternatives
            
            for alternative in alternatives:
                #print("Confidence: {}".format(alternative.confidence))
                #print(u"Transcript: {}".format(alternative.transcript))
                
                fhand.write("{}".format(alternative.transcript))
                #assembly.add(alternative.transcript)
                print(u"Transcript: {}".format(alternative.transcript))
                #print(assembly)
                #time.sleep(0.2)



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
            "source_language_code": "fr-FR",
            "target_language_code": "en-US",
        }
    )

    # Display the translation for each input text provided
    # Write transcription in a file 
    fhand = open("transcript.txt", "w")
    for translation in response.translations:
        fhand.write("{}".format(translation.translated_text))
        print("Translated text: {}".format(translation.translated_text))
    #time.sleep(0.5)


"""
fhand = open("transcript_file.txt", "r")
text = fhand.read()
translate_text(text, project_id="speechtotextapi-340414")
"""
#transcribe_streaming(os.path.abspath(audio_file))
#translate_text(text)
#th1 = threading.Thread(target=hah)
#th2 = threading.Thread(target=blabla)
"""
th1 = threading.Thread(target=lambda: translate_text(text))
th2 = threading.Thread(target=lambda: transcribe_streaming(os.path.abspath(audio_file)))

th1.start()
th2.start()

th1.join()
th2.join()
"""
print("HERE MUST START THE WORDCLOUD")
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
"""
file = os.path.abspath("transcript.txt")
t = frequency(file, english_stop_words)

fhand = open("transcript.txt", "r")
words = fhand.read()
wordcloud = WordCloud(background_color = 'white', stopwords = french_stop_words).generate(words)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
#transcribe_streaming(os.path.abspath(audio_file))
"""

"""
===============================================
===  DRAW THE WORDCLOUD OF THE TRANSLATION ====
===============================================
"""
def makeImage(text):
    wc = WordCloud(background_color="white")
    wc.generate_from_frequencies(text)
    
    plt.imshow(wc, interpolation = "bilinear")
    plt.axis("off")
    plt.show()


#compute the word error rate between the original
def wer(original, fromApi):
    fhand_original = open(original, "r")
    fhand_fromApi = open(fromApi, "r")
    or_sp = fhand_original.read().split()
    api_sp = fhand_fromApi.read().split()

    difference = list(set(or_sp).difference(api_sp))
    print("PRINTING THE DIFFERENCE")
    print(len(difference)/len(or_sp))
    plt.rcParams["figure.figsize"] = (11,8)
    plt.plot(or_sp, color="green", label='Original')
    
    plt.plot(api_sp, color="blue", label='Transcripted')
    #plt.plot(difference, color="black", label="Difference")
    #plt.fill_between(or_sp, api_sp, color="red", alpha=0.3)
    
    plt.title("Match between Original and Transcripted transcriptions")
    plt.xlabel("Transcription line number")
    plt.ylabel("MER (lower is better)")
    
    plt.legend()
    plt.show()


"""
===============================================
================ MAIN FUNCTION ================
===============================================
"""
def main():
    if __name__ == "__main__":
        transcribe_streaming(os.path.abspath(audio_file))
        fhand = open("transcript_file.txt", "r")
        text = fhand.read()
        print("Start of the translation")
        translate_text(text)


        
        file = os.path.abspath("transcript.txt")
        freq = frequency(file, english_stop_words)

        fhand = open("transcript.txt", "r")
        words = fhand.read()
        wordcloud = WordCloud(background_color = 'white', stopwords = french_stop_words).generate(words)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()

        makeImage(freq)

        print("COMPUTE THE WORD ERROR RATE")
        fromApi = os.path.abspath("transcript_file.txt")
        original = os.path.abspath("original.txt")
        wer(original, fromApi)

main()