# Vision Transformers and Compact Convolutional Transformers on CIFAR-10

## Overview

This repository compares two Transformer-based architectures on the CIFAR-10 dataset:

1. **Vision Transformer (ViT)**

   - **Patch Embedding**: The input image is split into fixed-size patches (e.g., 4×4). Each patch is linearly projected (using a `Conv2d` in our code) into a vector embedding.
   - **Positional Encoding**: Since Transformers are sequence-based, the patch embeddings are supplemented by positional encodings (either sinusoidal or learned) to preserve spatial information.
   - **Transformer Blocks**: Each block contains:
     1. **Multi-Head Self-Attention**: Splits the embedding into multiple heads, each head performs scaled dot-product attention, and the outputs are concatenated.
     2. **Layer Normalization**: Stabilizes training by normalizing across the feature dimension.
     3. **Feed-Forward Network (FFN)**: Typically a two-layer MLP with a non-linear activation.
   - **Classification Head**: A learnable `[CLS]` token is prepended, whose output is fed into an MLP for final classification.

2. **Compact Convolutional Transformer (CCT)**
   - **Convolutional Tokenizer**: Instead of slicing images into patches, CCT applies multiple convolution layers with pooling to create a sequence of “tokens.” This can help capture local spatial patterns early on.
   - **Transformer Blocks**: Similar to ViT, each block uses **Multi-Head Self-Attention** and an **FFN**.
   - **Sequence Pooling**: Uses a simple attention-based pooling layer (`SeqPool`) to summarize the token sequence into a single feature vector for classification.
   - **Classification Head**: The pooled token representation goes through a final MLP head to output class probabilities.

**What this project does**:

- Loads CIFAR-10 data.
- Builds the **ViT** and **CCT** models from smaller modules (attention layers, FFNs, etc.).
- Trains both models, logging train/test loss and accuracy.
- Implements a **checkpoint mechanism** to save and resume training seamlessly.
- Plots the loss and accuracy curves to compare both architectures.

---

## Installation

1. **Clone** this repository:

   ```bash
   git clone https://github.com/TristanDonze/VisionTransformer.git
   cd VisionTransformer
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run Jupyter Notebook** to train/evaluate the models:

   ```bash
   jupyter notebook main.ipynb
   ```

4. **Execute all cells** in the notebook to train and compare the ViT and CCT models.

   It will load automatically pretrained models. To train the models from scratch, simply delete the `models` folder.
