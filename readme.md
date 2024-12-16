# Phishing URL Checker

A machine learning-based tool to detect phishing and malicious URLs.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=flat&logo=Keras&logoColor=white)](https://keras.io)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/li/Phishing-URL-Checker/graphs/commit-activity)

## Description

This project uses deep learning to classify URLs as either safe or malicious. It employs a convolutional neural network model trained on a large dataset of labeled URLs to identify potential phishing attempts and malware distribution sites.

## Features

- URL safety classification using deep learning
- Real-time URL checking functionality
- Simple GUI interface for URL input
- Command line interface for batch processing
- Supports both single URL and text with multiple URLs
- Returns probability scores for malicious classification

## Technologies Used

- Python 
- TensorFlow/Keras
- Tkinter (for GUI)
- NumPy
- Pandas

## Model Details

- Input: URLs encoded as sequences of characters
- Architecture: Convolutional Neural Network with 3 convolutional layers
- Output: Binary classification (Safe/Malicious) with probability score
- Training accuracy: ~95%
- Validation accuracy: ~94%

## Usage

### NoteBook
```python
python URL_Checker.ipynb
```
Enter a URL in the input field and click "Check URL" to get the classification result.

### Command Line Interface
```python
python tool.py
```
Input text containing URLs when prompted. The script will extract and check all URLs found.

## Installation

1. Clone this repository
2. Install required packages:
```bash 
pip install tensorflow numpy pandas string
```
3. Download the pre-trained model file (`model_40.keras`)
4. Run either the GUI or command line interface

## Project Structure

- `URL_Checker.ipynb` - Main notebook
- `test2.py` - Command line interface
- `models/` - Directory containing trained model
- `dataset/` - Training data (CSV format)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Dataset and base architecture inspired by various phishing detection research papers and implementations.