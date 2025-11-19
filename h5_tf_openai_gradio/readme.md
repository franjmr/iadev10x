# GPC H5 Model with OpenAI and Gradio
This project demonstrates how to use a pre-trained TensorFlow model saved in H5 format to classify graphics cards based on their images. The model predicts the manufacturer of the graphics card from a set of predefined classes and ask OpenAI for additional information about the predicted class. 
A Gradio interface is provided for easy interaction with the model.

## Features
- Load a pre-trained TensorFlow model from an H5 file.
- Use Gradio to create a user-friendly web interface for image upload and classification.
- Integrate OpenAI API to fetch additional information about the predicted graphics card manufacturer.
- Display results in an interactive web app.
- Chat with the model to get more insights.

## Prerequisites
- tensorflow
- Pillow
- numpy
- OpenAI
- python-dotenv
- scikit-learn
- gradio

## Installation Instructions
### Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
``` 
### Install required packages
```bash
pip install -r requirements.txt
# Or install manually
pip install tensorflow pillow numpy
```
### Deactivate the virtual environment when done
```bash
deactivate
```