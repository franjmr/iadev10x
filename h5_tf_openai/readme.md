# GPC H5 Model
This project demonstrates how to use a pre-trained TensorFlow model saved in H5 format to classify graphics cards based on their images. The model predicts the manufacturer of the graphics card from a set of predefined classes and ask OpenAI for additional information about the predicted class.

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