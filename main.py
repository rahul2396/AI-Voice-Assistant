# main changes in new code - google gemini , amazon polly rest more or less same (functionality wise)

import os
from os import PathLike
from time import time
import asyncio
from typing import Union
from langdetect import detect, LangDetectException
from dotenv import load_dotenv, dotenv_values
import google.generativeai as genai
from deepgram import Deepgram
import pygame
from pygame import mixer
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from contextlib import closing
import sys
import sounddevice as sd
from scipy.io import wavfile
import numpy as np
import deepl

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the .env file
dotenv_path = os.path.join(script_dir, '.env')

# Load environment variables
load_dotenv(dotenv_path)
config = dotenv_values(dotenv_path)
print(f"Contents of .env file: {config}")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")

# Initialize APIs
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')
deepgram = Deepgram(DEEPGRAM_API_KEY)
translator = deepl.Translator(DEEPL_API_KEY)

# Initialize Amazon Polly client
polly_client = boto3.Session(region_name="ap-south-1").client('polly')  # Using India (Mumbai) region

# Initialize pygame mixer for MP3
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)

# Change the context if you want to change Jarvis' personality
context = "You are Jarvis, Alex's human assistant. Your answers should be limited to 1-2 short sentences."
RECORDING_PATH = "audio/recording.wav"
RESPONSE_PATH = "audio/response.mp3"
SAMPLE_RATE = 44100  # Standard sample rate
DURATION = 5  # Record for 5 seconds

def log(message: str):
    """Print and write to status.txt"""
    print(message)
    with open("status.txt", "w") as f:
        f.write(message)

def speech_to_text():
    """Record audio and save it to a file."""
    log("Recording...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()  # Wait until recording is finished
    log("Recording finished")
    
    # Normalize the recording to 16-bit range
    recording = np.int16(recording / np.max(np.abs(recording)) * 32767)
    
    # Save as WAV file
    wavfile.write(RECORDING_PATH, SAMPLE_RATE, recording)
    log(f"Audio saved to {RECORDING_PATH}")

async def transcribe(file_name: Union[Union[str, bytes, PathLike[str], PathLike[bytes]], int]):
    """Transcribe audio using Deepgram API."""
    with open(file_name, "rb") as audio:
        source = {"buffer": audio, "mimetype": "audio/wav"}
        response = await deepgram.transcription.prerecorded(source)
        return response["results"]["channels"][0]["alternatives"][0]["words"]

def request_gemini(prompt: str) -> str:
    """Send a prompt to the Gemini API and return the response."""
    response = model.generate_content(prompt)
    return response.text

def translate_text(text, target_language='EN-US'):
    """Translate text to the target language using DeepL."""
    try:
        result = translator.translate_text(text, target_lang=target_language)
        return str(result)
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

def generate_audio(text, output_file, language='en-US'):
    """Generate audio using Amazon Polly"""
    try:
        voice_id = "Joanna"  # Default English voice
        if language.startswith('fr'):
            voice_id = "Celine"  # French voice
        elif language.startswith('de'):
            voice_id = "Marlene"  # German voice
        # Add more language-voice mappings as needed

        response = polly_client.synthesize_speech(Text=text, OutputFormat="mp3", VoiceId=voice_id)
        if "AudioStream" in response:
            with closing(response["AudioStream"]) as stream:
                try:
                    with open(output_file, "wb") as file:
                        file.write(stream.read())
                except IOError as error:
                    print(error)
                    sys.exit(-1)
        else:
            print("Could not stream audio")
            sys.exit(-1)
    except (BotoCoreError, ClientError) as error:
        print(error)
        sys.exit(-1)

if __name__ == "__main__":
    user_language = 'EN-US'  # Default language, can be changed based on user preference
    while True:
        # Record audio
        log("Listening...")
        speech_to_text()
        log("Done listening")

        # Transcribe audio
        current_time = time()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        words = loop.run_until_complete(transcribe(RECORDING_PATH))
        string_words = " ".join(word_dict.get("word") for word_dict in words if "word" in word_dict)
        
        if not string_words.strip():
            log("No speech detected. Please try again.")
            continue

        with open("conv.txt", "a") as f:
            f.write(f"User: {string_words}\n")
        transcription_time = time() - current_time
        log(f"Finished transcribing in {transcription_time:.2f} seconds.")

        # Detect language and translate if necessary
        try:
            detected_language = detect(string_words)
            if detected_language != 'en':
                # Translate to English for processing
                english_input = translate_text(string_words, 'EN-US')
                log(f"Translated input: {english_input}")
            else:
                english_input = string_words
        except LangDetectException:
            english_input = string_words  # Default to original input if detection fails

        # Get response from Gemini
        current_time = time()
        context += f"\nAlex: {english_input}\nJarvis: "
        response = request_gemini(context)
        context += response
        gemini_time = time() - current_time
        log(f"Finished generating response in {gemini_time:.2f} seconds.")

        # Translate response back to user's language if necessary
        if detected_language != 'en':
            response = translate_text(response, user_language)

        # Generate audio response using Amazon Polly
        current_time = time()
        generate_audio(response, RESPONSE_PATH, user_language)
        audio_time = time() - current_time
        log(f"Finished generating audio in {audio_time:.2f} seconds.")

        # Play response
        log("Speaking...")
        sound = pygame.mixer.Sound(RESPONSE_PATH)
        with open("conv.txt", "a") as f:
            f.write(f"Jarvis: {response}\n")
        sound.play()
        pygame.time.wait(int(sound.get_length() * 1000))
        print(f"\n --- USER ({detected_language}): {string_words}")
        print(f" --- JARVIS ({user_language}): {response}\n")

    #end

#     """Main file for the Jarvis project"""
# import os
# from os import PathLike
# from time import time
# import asyncio
# from typing import Union

# from dotenv import load_dotenv
# import openai
# from deepgram import Deepgram
# import pygame
# from pygame import mixer
# import elevenlabs

# from record import speech_to_text

# # Load API keys
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
# elevenlabs.set_api_key(os.getenv("ELEVENLABS_API_KEY"))

# # Initialize APIs
# gpt_client = openai.Client(api_key=OPENAI_API_KEY)
# deepgram = Deepgram(DEEPGRAM_API_KEY)
# # mixer is a pygame module for playing audio
# mixer.init()

# # Change the context if you want to change Jarvis' personality
# context = "You are Jarvis, Alex's human assistant. You are witty and full of personality. Your answers should be limited to 1-2 short sentences."
# conversation = {"Conversation": []}
# RECORDING_PATH = "audio/recording.wav"


# def request_gpt(prompt: str) -> str:
#     """
#     Send a prompt to the GPT-3 API and return the response.

#     Args:
#         - state: The current state of the app.
#         - prompt: The prompt to send to the API.

#     Returns:
#         The response from the API.
#     """
#     response = gpt_client.chat.completions.create(
#         messages=[
#             {
#                 "role": "user",
#                 "content": f"{prompt}",
#             }
#         ],
#         model="gpt-3.5-turbo",
#     )
#     return response.choices[0].message.content


# async def transcribe(
#     file_name: Union[Union[str, bytes, PathLike[str], PathLike[bytes]], int]
# ):
#     """
#     Transcribe audio using Deepgram API.

#     Args:
#         - file_name: The name of the file to transcribe.

#     Returns:
#         The response from the API.
#     """
#     with open(file_name, "rb") as audio:
#         source = {"buffer": audio, "mimetype": "audio/wav"}
#         response = await deepgram.transcription.prerecorded(source)
#         return response["results"]["channels"][0]["alternatives"][0]["words"] - channels for multiple audio source (like multiple mics), words - the word with most confidence is genereall the first in list


# def log(log: str):
#     """
#     Print and write to status.txt
#     """
#     print(log)
#     with open("status.txt", "w") as f:
#         f.write(log)


# if __name__ == "__main__":
#     while True:
#         # Record audio
#         log("Listening...")
#        speech_to_text()
#         log("Done listening")

#         # Transcribe audio
#         current_time = time()
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         words = loop.run_until_complete(transcribe(RECORDING_PATH))
#         string_words = " ".join(
#             word_dict.get("word") for word_dict in words if "word" in word_dict- dict(data structure to store value) - takes only words if they have a key so like hello world is taken and timestamp:1.0 is not
#         )
#         with open("conv.txt", "a") as f:
#             f.write(f"{string_words}\n")
#         transcription_time = time() - current_time
#         log(f"Finished transcribing in {transcription_time:.2f} seconds.")

#         # Get response from GPT-3
#         current_time = time()
#         context += f"\nAlex: {string_words}\nJarvis: "
#         response = request_gpt(context)
#         context += response - prev response added to context to maintain conversation
#         gpt_time = time() - current_time
#         log(f"Finished generating response in {gpt_time:.2f} seconds.")

#         # Convert response to audio
#         current_time = time()
#         audio = elevenlabs.generate(
#             text=response, voice="Adam", model="eleven_monolingual_v1"
#         )
#         elevenlabs.save(audio, "audio/response.wav")
#         audio_time = time() - current_time
#         log(f"Finished generating audio in {audio_time:.2f} seconds.")

#         # Play response
#         log("Speaking...")
#         sound = mixer.Sound("audio/response.wav")
#         # Add response as a new line to conv.txt
#         with open("conv.txt", "a") as f:
#             f.write(f"{response}\n")
#         sound.play()
#         pygame.time.wait(int(sound.get_length() * 1000)) - loop doesnt record when response is playing
#         print(f"\n --- USER: {string_words}\n --- JARVIS: {response}\n")

# for translation-
#lang detect , google cloud translate v2 , 