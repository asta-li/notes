"""
Load and process audio data from audio and video recordings.
"""

import os

import ffmpeg
import librosa
import soundfile


def convert_audio_format(file_path: str) -> str:
    """Convert audio file to '.wav' format."""
    OUTPUT_FORMAT = '.wav'
    name, ext = os.path.splitext(file_path)
    if ext == OUTPUT_FORMAT:
        return file_path

    output_path = name + OUTPUT_FORMAT
    stream = ffmpeg.input(file_path)
    stream = ffmpeg.output(stream, output_path)
    ffmpeg.run(stream)
    return output_path


def load_audio_data(file_path: str) -> (list, int):
    """Load audio data from video or audio file.

    Returns:
        Tuple containing the audio data and sample rate.
    """
    # TODO(asta): Support video recordings. This currently only supports audio.
    # Soundfile supports only libsndfile-supported formats: WAV, FLAC, OGG, MAT
    SUPPORTED_FORMATS = ('.wav', '.m4a')
    if not file_path.endswith(SUPPORTED_FORMATS):
        raise ValueError('Only {} file formats are supported!'.format(SUPPORTED_FORMATS), file_path)

    file_path = convert_audio_format(file_path)

    return soundfile.read(file_path)