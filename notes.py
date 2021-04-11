"""
Transcribe video recordings (e.g. from Zoom or Google Meet) and upload to Notion.
"""

import argparse
import logging
import os
import sys
import time

import scribe
import upload

logger = logging.getLogger(__name__)


def transcribe(file_paths: list):
    """Transcribe audio and video recordings (e.g. from Zoom or Google Meet) and upload to Notion.
    
    Args:
        file_paths: List of paths to audio or video recording files.
    """
    logger.info('Initializing...')
    transcriber = scribe.Transcriber()
    summarizer = scribe.Summarizer()
    uploader = upload.Uploader()

    for file_path in file_paths:
        logger.info('Transcribing: %s', file_path)
        transcription = transcriber.transcribe(file_path)
        summary = summarizer.summarize(transcription)

        logger.info('Summary: %s', summary)
        logger.info('Transcription: %s', transcription)

        # TODO(asta): Pull meeting participants from Google calendar.
        datetime = time.ctime(os.path.getctime(file_path))
        title = 'Meeting notes {}'.format(datetime)

        uploader.upload(title, summary, transcription, datetime)
        logger.info('Uploaded transcription to Notion!')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    input_parser = parser.add_mutually_exclusive_group()
    input_parser.add_argument(
        '--dir', # default='/Users/{}/Documents/Zoom/'.format(os.getlogin()),
        help='Input directory containing audio or video recordings.')
    input_parser.add_argument(
        '--file',
        help='Path to an audio or video recording.')
    args = parser.parse_args()
    
    # Load file paths.
    if args.dir:
        files = [os.path.join(args.dir, file_name) for file_name in os.listdir(args.dir)]
    else:
        files = [args.file]

    # Transcribe recordings and upload.
    transcribe(files)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())