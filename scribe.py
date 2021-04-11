"""
Transcribe English audio to text.
"""

import os
import sys

import librosa
import nltk
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, T5Tokenizer, T5ForConditionalGeneration

import audio 


class Summarizer:
    """Summarize text."""
    def __init__(self):
        # Use Google's T5 base model: https://huggingface.co/t5-base
        PRETRAINED_MODEL = 't5-base'
        self.model = T5ForConditionalGeneration.from_pretrained(PRETRAINED_MODEL)
        self.tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_MODEL)

    def summarize(self, text: str) -> str:
        """Summarize the input text as a short snippet (<50 tokens)."""
        # Prepend with "summarize: ", since T5 is a general-purpose language model.
        # Encode the input text. Note we truncate if the input is longer than 512 tokens.
        inputs = self.tokenizer.encode(
            'summarize: ' + text, return_tensors='pt', max_length=512, truncation=True)
        
        MIN_LENGTH = 50
        MAX_LENGTH = 150
        assert MIN_LENGTH < MAX_LENGTH
        if inputs.shape[1] < MIN_LENGTH:
            return text

        # See docs at https://huggingface.co/transformers/v2.9.1/main_classes/model.html#transformers.PreTrainedModel.generate
        outputs = self.model.generate(
            inputs, 
            max_length=MAX_LENGTH, 
            min_length=MIN_LENGTH,
            length_penalty=2.0, 
            num_beams=5,
            early_stopping=True)
        raw_summary = self.tokenizer.decode(outputs[0])

        # Correct summary formatting.
        formatted_summary = []
        for sentence in raw_summary.split('. '):
            # Strip padding and whitespace.
            sentence = sentence.strip('<pad>').strip('.</s>').strip()
            formatted_summary.append(sentence[0].capitalize() + sentence[1:] + '.')
        return ' '.join(formatted_summary)


class Transcriber:
    """Transcribe audio data to text."""
    def __init__(self):
        self.REQUIRED_SAMPLE_RATE = 16000

        # Use Facebook's pretrained Wav2Vec2 model
        # https://huggingface.co/facebook/wav2vec2-large-960h
        PRETRAINED_MODEL = 'facebook/wav2vec2-base-960h'
        self.processor = Wav2Vec2Processor.from_pretrained(PRETRAINED_MODEL)
        self.model = Wav2Vec2ForCTC.from_pretrained(PRETRAINED_MODEL)


    def transcribe(self, file_path: str) -> str:
        """Transcribe audio from a file to text."""
        # TODO(asta): Stream over smaller chunks instead of loading the full file.
        # stream = librosa.stream(file_path, block_length=30, frame_length=16000, hop_length=16000)
        # for audio_data in stream: ...

        # Load the audio data from file.
        audio_data, sample_rate = audio.load_audio_data(file_path)

        # Transcribe the audio chunk.
        transcription = self.__transcribe_chunk(audio_data, sample_rate)
        return self.__format_sentences(transcription)


    def __transcribe_chunk(self, audio_data: list, sample_rate: int) -> str:
        """Transcribe audio data to text."""
        # Wav2Vec2 was trained on audio sampled at 16Khz.
        if sample_rate != self.REQUIRED_SAMPLE_RATE:
            audio_data = librosa.resample(audio_data, sample_rate, self.REQUIRED_SAMPLE_RATE)

        # Preprocess audio data and run transcription model.
        input_values = self.processor(
            audio_data, sampling_rate=self.REQUIRED_SAMPLE_RATE, return_tensors='pt').input_values
        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription.lower()


    def __format_sentences(self, text: str) -> str:
        """Break and format text into sentences."""
        sentences = nltk.sent_tokenize(text)
        # Capitalize the first word in each sentence.
        sentences = [s.replace(s[0], s[0].capitalize(), 1) + '.' for s in sentences]
        return ' '.join(sentences)

