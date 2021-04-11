# Automatic notetaking in Notion for video calls

## Enter the conda environment
conda activate notes

## Requirements
- Install miniconda: https://docs.conda.io/en/latest/miniconda.html
- Set up the conda and pip environments.
```
conda env create -f environment.yml
pip install -r requirements.txt
```

## Run the code
Run `notes.py` on an audio file or directory containing a list of files.
```
python notes.py --file $FILE_PATH  
```

## Exit the conda environment
conda deactivate
