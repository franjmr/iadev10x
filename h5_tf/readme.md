# GPC H5 Model
This project demonstrates how to use a pre-trained TensorFlow model saved in H5 format to classify graphics cards based on their images. The model predicts the manufacturer of the graphics card from a set of predefined classes.

## Model Classes
The model classifies graphics cards into the following categories:
- Class 0: 3dfx
- Class 1: ATI
- Class 2: Matrox
- Class 3: NVIDIA
- Class 4: S3
- Class 5: Trident

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

## Usage
1. Place the images of the graphics cards you want to classify in the `_assets` folder
2. Run the `main.py` script to classify the images and see the results:
```bash
python main.py
```
3. The script will output the predicted class for each image along with the overall accuracy of the model.

## Notes
- Ensure that the images are in a compatible format (e.g., JPG, PNG) and are preprocessed correctly before feeding them into the model.
- Modify the `cards` list in `main.py` to include paths to your own images for classification.