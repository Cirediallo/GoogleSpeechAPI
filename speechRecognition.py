import os
#import gcloud client library
from google.cloud import speech


credential_path = "C:\\Users\Mamadou\Documents\Cours\M1 ATAL\S2\TER\Google Speech API\speechtotextapi-340414-1e2134d625a1.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

client = speech.SpeechClient()

# The name of the audio to transcribe
gcs_uri = "gs://ter_bucket/OSR_us_000_0010_8k.wav"

audio = speech.RecognitionAudio(uri=gcs_uri)

config = speech.RecognitionConfig(
    encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz = 8000, # 16000 won't work use 8000
    language_code = "en-US",
)

#Detects speech in the audio
response = client.recognize(config=config, audio=audio)

print(response)
"""
response is an object represented as:
response{
    alternatives{
        transcript: "here a portion of the transcription"
        confidence: here_a_time
    }
    result_end_time{
        seconds: here_a_number
        nanos: here_a_number
    }
    language_code: "the_language_code_provided_in_source_code"
}
"""

for result in response.results:
    print("Transcript: {}".format(result.alternatives[0].transcript))

