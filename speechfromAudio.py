import os
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
        language_code="fr-FR", #change en-US to fr-FR
    )

    streaming_config = speech.StreamingRecognitionConfig(config=config)

    # streaming_recognize returns a generator.
    responses = client.streaming_recognize(
        config=streaming_config,
        requests=requests,
    )

    for response in responses:
        # Once the transcription has settled, the first result will contain the
        # is_final result. The other results will be for subsequent portions of
        # the audio.
        for result in response.results:
            print("Finished: {}".format(result.is_final))
            print("Stability: {}".format(result.stability))
            alternatives = result.alternatives
            # The alternatives are ordered from most likely to least.
            for alternative in alternatives:
                print("Confidence: {}".format(alternative.confidence))
                print(u"Transcript: {}".format(alternative.transcript))



stream_file = "C:\\Users\Mamadou\Documents\Cours\M1 ATAL\S2\TER\Google Speech API\OSR_fr_000_0041_8k.wav"
transcribe_streaming(stream_file)