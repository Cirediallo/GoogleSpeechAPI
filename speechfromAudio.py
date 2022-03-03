import os
import time
import threading

credential_path = "C:\\Users\Mamadou\Documents\Cours\M1 ATAL\S2\TER\Google Speech API\speechtotextapi-340414-1e2134d625a1.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

def transcribe_streaming(stream_file):
    """Streams transcription of the given audio file."""
    import io
    from google.cloud import speech

    
    client = speech.SpeechClient()



    with io.open(stream_file, "rb") as audio_file:
        content = audio_file.read()

    
    # In practice, stream should be a generator yielding chunks of audio data.
    stream = [content]

    requests = (
        speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in stream
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=8000,
        language_code="en-US", #change en-US to fr-FR
    )

    streaming_config = speech.StreamingRecognitionConfig(config=config)

    # streaming_recognize returns a generator.
    responses = client.streaming_recognize(
        config=streaming_config,
        requests=requests,
    )

    #create a file where to put transctiption
    fhand = open('local_audio_transcript.txt', 'w')
    for response in responses:
        # Once the transcription has settled, the first result will contain the
        # is_final result. The other results will be for subsequent portions of
        # the audio.
        for result in response.results:
            #print("Finished: {}".format(result.is_final))
            #print("Stability: {}".format(result.stability))
            alternatives = result.alternatives
            # The alternatives are ordered from most likely to least.
            for alternative in alternatives:
                #print("Confidence: {}".format(alternative.confidence))
                print(u"Transcript: {}".format(alternative.transcript))
                
                fhand.write("{}".format(alternative.transcript))
                






stream_file = "C:\\Users\Mamadou\Documents\Cours\M1 ATAL\S2\TER\Google Speech API\OSR_us_000_0010_8k.wav"
#transcribe_streaming(stream_file)

""" TRANSCRIPTION """

# Imports the Google Cloud Translation library
from google.cloud import translate

# Initialize Translation client
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
            "target_language_code": "fr",
        }
    )

    # Display the translation for each input text provided
    for translation in response.translations:
        print("Translated text: {}".format(translation.translated_text))
        time.sleep(0.5)


fhand = open('local_audio_transcript.txt', 'r')
# we read the whole file because it's a small file 
# in case of big files modify the writing option first
# then here the reading option
text = fhand.read() 
#translate_text(text)

th_transciption = threading.Thread(transcribe_streaming(stream_file))
th_translation = threading.Thread(translate_text(text))