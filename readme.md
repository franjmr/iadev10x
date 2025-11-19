# IA Dev 10x
Repository with different implementations of IA workshops using different technologies:
- Transfer Learning with ImageNet
- TensorFlow/Keras for image classification
- Integration with OpenAI APIs
- Gradio for graphical user interface

## Projects

### h5_imagenet_transfer_learning
Base implementation of Transfer Learning using ImageNet. Contains:
- Jupyter Notebook (`3d3rdgen_imagenets.ipynb`) with model training
- Trained model (`model_graphic_cards_transfer_learning.h5`) for graphics card classification

### h5_tf
TensorFlow/Keras model implementation that allows performing inferences on images.
- Main script (`main.py`) to run predictions
- `_assets` folder to store test images
- `_model` folder with training notebook and compiled model

### h5_tf_openai
Evolution of the previous project integrating OpenAI capabilities.
- Maintains image classification functionality with TensorFlow
- Training script available in both notebook and Python formats (`3d3rdgen_imagenets.py`)
- Integration with OpenAI APIs for additional processing

### h5_tf_openai_gradio
Version with graphical user interface using Gradio.
- Provides an interactive web interface to upload images and get predictions
- Combines the classification model with OpenAI capabilities
- Easy to use without code, ideal for demonstrations

## Requirements
Each project includes its own `requirements.txt` file with the necessary dependencies.

## Model
All projects use the same base model trained with Transfer Learning for graphics card classification.