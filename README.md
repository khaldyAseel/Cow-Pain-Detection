

# Cow Pain Detection Model

This repository contains a machine learning project designed to detect pain or no pain in cows based on images of their faces and bodies. The model utilizes Convolutional Neural Networks (CNN), Visual Transformers, and Multi-Layer Perceptron (MLP) architectures to classify cow images into two categories: `Pain` and `No Pain`.

## Dataset

The dataset consists of images of cows' faces and bodies. Each image is labeled with either `Pain` or `No Pain` based on the cow’s physical state. The dataset is split into training, validation, and test sets.

### Data Organization

- `cow_face`: Folder containing images of cows' faces.
- `cow_body`: Folder containing images of cows' bodies.

The images are pre-processed and resized to `224x224` pixels to be fed into the models.

## Requirements

To run this project, you need the following Python libraries:

- torch
- torchvision
- scikit-learn
- tqdm
- numpy
- matplotlib

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Code Overview

### Data Preprocessing
The dataset is divided into training, validation, and test sets. The images are resized to `224x224` pixels and normalized before being passed to the models.

### Model Architectures
This project utilizes the following models:
- **CNN (Convolutional Neural Network)**: A basic CNN with three convolutional layers followed by fully connected layers for classification.
- **Visual Transformer**: (Add description of transformer model here if used).
- **MLP (Multi-Layer Perceptron)**: A simple fully connected neural network used for classification.

### Training and Evaluation
- The model is trained on both cow face and body images.
- Metrics including loss, accuracy, precision, recall, and F1 score are reported during each epoch.
- The final evaluation metrics are calculated on the test dataset.

### Training Command

To train the model, run the following command:

```bash
python train.py
```

The training script will train the model for 10 epochs (can be adjusted) and print training and validation metrics.

## Results

After training, the model's performance is evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

The metrics are calculated for both the training and validation datasets at each epoch.

## Future Work
- Fine-tuning and testing different model architectures.
- Using data augmentation techniques to improve model robustness.
- Incorporating additional data sources for improved accuracy.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

