import os
import wave
import pyaudio
import numpy as np
from scipy.io import wavfile
from openai import OpenAI
import voice_service as vs
from rag.AIVoiceAssistant import AIVoiceAssistant
from langdetect import detect  # For language detection
from deep_translator import GoogleTranslator  # For translation

# Initialize OpenAI client
client = OpenAI(api_key="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")

# Initialize AI Assistant
ai_assistant = AIVoiceAssistant()

# Constants
DEFAULT_CHUNK_LENGTH = 5  # Reduced chunk length to prevent overflow

def is_silence(data, max_amplitude_threshold=3000):
    """Check if audio data contains silence."""
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold

def record_audio_chunk(audio, stream, chunk_length=DEFAULT_CHUNK_LENGTH):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        try:
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        except OSError as e:
            print(f"Error reading audio data: {e}")
            return False

    temp_file_path = 'temp_audio_chunk.wav'
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    # Check if the recorded chunk contains silence
    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data):
            os.remove(temp_file_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error while reading audio file: {e}")
        return False

def transcribe_audio(audio_file_path):
    """Transcribe audio using OpenAI's Whisper."""
    with open(audio_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcription.text

def detect_language(text):
    """Detect the language of the input text."""
    try:
        lang = detect(text)
        return lang
    except:
        return "en"  # Default to English if detection fails

def translate_text(text, src_lang, dest_lang="en"):
    """Translate text from source language to destination language."""
    try:
        translated = GoogleTranslator(source=src_lang, target=dest_lang).translate(text)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

def generate_response(input_text):
    """Generate a response using the AI assistant."""
    response = ai_assistant.interact_with_llm(input_text)
    return response.lstrip() if response else ""

def speak_response(text, language="en"):
    """Speak the response in the specified language."""
    vs.play_text_to_speech(text, language=language)

def main():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    customer_input_transcription = ""

    try:
        while True:
            chunk_file = "temp_audio_chunk.wav"
            
            # Record audio chunk
            print("_")
            if not record_audio_chunk(audio, stream):
                # Transcribe audio using OpenAI's Whisper
                transcription = transcribe_audio(chunk_file)
                os.remove(chunk_file)
                print("Customer: {}".format(transcription))
                
                # Detect the language of the input
                input_lang = detect_language(transcription)
                print(f"Detected language: {input_lang}")
                
                # Translate Marathi input to English
                if input_lang == "mr":  # Marathi
                    english_input = translate_text(transcription, src_lang="mr", dest_lang="en")
                    print(f"Translated to English: {english_input}")
                else:
                    english_input = transcription  # Use as-is for English
                
                # Generate response in English
                english_response = generate_response(english_input)
                print(f"AI Assistant (English): {english_response}")
                
                # Translate response back to Marathi if input was in Marathi
                if input_lang == "mr":
                    marathi_response = translate_text(english_response, src_lang="en", dest_lang="mr")
                    print(f"AI Assistant (Marathi): {marathi_response}")
                    speak_response(marathi_response, language="mr")  # Speak in Marathi
                else:
                    print(f"AI Assistant (English): {english_response}")
                    speak_response(english_response, language="en")  # Speak in English

    except KeyboardInterrupt:
        print("\nStopping...")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if stream.is_active():
            stream.stop_stream()
        if stream.is_open():
            stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()