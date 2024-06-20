
<body>
    <h1>Deep Learning Projects (Transportation objects classification)</h1>

## Overview

This project focuses on training neural network models to recognize 12 different types of transportation objects from the xView database, which contains annotated images from a zenithal perspective. The primary objective is to classify objects in satellite images using various neural network architectures, including Feed-Forward Neural Networks (FFNN), Convolutional Neural Networks (CNN), and Transfer Learning models.

## Process

### Data Preparation
- The dataset comprises 796 annotated satellite images, split into 44963 smaller images of 224x224 pixels.
- These images were divided into training (90%) and test (10%) sets.

### Model Development
#### Feed-Forward Neural Networks (FFNN)
- Started with a simple architecture: Flatten layer, Dense layer with 20% Dropout, 'Gelu' activation function, and Softmax output layer.
- Gradually added Batch Normalization, increased Dropout to 40%, and tested different optimizers (Adam, Adamax).
- Best model architecture: 
  - Flatten layer
  - Dense layer with 128 neurons, 'Gelu' activation
  - Batch Normalization
  - Dropout (20%)
  - Output layer with Softmax activation
- Achieved an accuracy of 41.88% and a recall of 23.28%.

#### Convolutional Neural Networks (CNN)
- Initial architecture: Three convolutional layers with 32 and 64 filters, followed by Max Pooling and a Flatten layer.
- Added Batch Normalization and experimented with different dropout rates (20% and 40%).
- Best model architecture (CNN_8):
  - Three convolutional layers (32 filters, 64 filters, 64 filters)
  - Max Pooling layers
  - Flatten layer
  - Dense layer with 256 neurons, 'ReLU' activation
  - Dense layer with 128 neurons, 'ReLU' activation
  - Dropout (20%)
  - Output layer with Softmax activation
- Achieved an accuracy of 68.97% and a recall of 36.65%.

#### Transfer Learning
- Used pre-trained models: VGG19, ResNet50, Xception.
- Modified top layers with Flatten, Dense layers, and Dropout (20%) to adapt to the new dataset.
- Best model: VGG19 with non-frozen layers.
  - Flatten layer
  - Dense layer with 256 neurons, 'ReLU' activation
  - Dropout (20%)
  - Dense layer with 128 neurons, 'ReLU' activation
  - Output layer with Softmax activation
- Achieved an accuracy of 77.05%.

### Training and Optimization
- Models were trained using Google Colab, leveraging both free and Pro versions to handle computational requirements.
- Optimizations included the use of Batch Normalization and Dropout to prevent overfitting.

## Results

### Best Models
- **FFNN:** The best-performing model incorporated Batch Normalization and achieved an accuracy of 41.88%.
- **CNN:** The top model (CNN_8) achieved an accuracy of 68.97% with additional Dense layers.
- **Transfer Learning:** The VGG19 model trained from scratch showed the highest accuracy of 77.05%.

### Key Findings
- Batch Normalization improved FFNN performance but not CNN performance.
- Dropout effectively prevented overfitting.
- Data imbalance was a significant challenge, particularly for the Helicopter class, which could benefit from Data Augmentation.

## Conclusion
The project successfully demonstrated the application of various neural network architectures to classify transportation objects in satellite images. The findings highlight the importance of architecture choice and regularization techniques in achieving optimal performance.

  
 
