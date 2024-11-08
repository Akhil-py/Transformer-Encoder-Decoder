# Transformer Model Implementation

## Overview

This project implements a GPT-like transformer model consisting of an encoder and a decoder for various natural language processing tasks, including text classification and language modeling. The architecture is designed to leverage the power of self-attention mechanisms to understand and generate human-like text. Specifically, the model processes speeches from U.S. Presidents Barack Obama, George H.W. Bush, and George W. Bush, learning to classify speech segments and generate text that mimics their individual speaking styles. The encoder component focuses on text classification, while the decoder is used for language modeling tasks, allowing the system to both analyze and generate presidential-style speech patterns.

## Features

- **Encoder**: A multi-layer transformer encoder that processes input sequences and generates contextual embeddings.
- **Classifier**: A feedforward neural network that takes the encoder's output and predicts class labels for text classification tasks.
- **Decoder**: An autoregressive transformer decoder that predicts the next token in a sequence based on previous tokens.
- **Perplexity Evaluation**: Computes perplexity as a measure of how well the model predicts the next token in a sequence.

## File Structure

- `transformer.py`: Contains the implementation of the transformer encoder, decoder, and related components.
- `main.py`: The main script for training the model, evaluating performance, and running inference.
- `tokenizer.py`: Implements a simple tokenizer for converting text data into numerical format.
- `dataset.py`: Contains dataset classes for loading and preprocessing data for classification and language modeling tasks.
- `utilities.py`: Helper functions for visualization and data processing.



## Setup and Installation
1. Uses Python 3.x
2. Install the following libraries
```
pip install torch
pip install torch nltk
```
3. For the initial run, uncomment the nltk.download statements in `tokenizer.py`
4. To run the Encoder and Decoder Transformer Blocks, simply run
```python main.py
```
5. To run only the encoder and classifier block, run:
```python main.py -part1
```
6. To run only the decoder, run:
```python main.py -part2
```

## Part 1: Encoder with Classifier

The Encoder with Classifier will run first and its Train and Test Accuracy will be reported for each epoch (15 total). Accuracies are recorded to be over 80% after 15 epochs. An attention map will be displayed at the end. Close it to move onto the Decoder implementation below.

## Part 2: Decoder Language Model

_If running both part1 and part2 together: Once Part 1 has run and the attention map is displayed, close it to start running the Decoder._

The Decoder will run for 500 epochs and report its perplexity scores on the train, hbush, obama, and wbush text files.
Perplexity scores for the training file is recorded to be in the mid to high 100s. The rest are recorded in the 300s and 400s.
Finally, the number of parameters in the decoder is printed out.

## Hyperparameters

The following hyperparameters can be adjusted in `main.py`:
- batch_size: Number of sequences processed in parallel (default: 16)
- block_size: Maximum context length for predictions (default: 32)
- learning_rate: Learning rate for the optimizer (default: 1e-3)
- n_embd: Embedding dimension (default: 64)
- n_head: Number of attention heads (default: 2)
- n_layer: Number of transformer layers (default: 4)
- epochs_CLS: Number of epochs for classifier training (default: 15)

## Results

The implementation allows you to monitor training progress, including loss values and perplexity after specified intervals. Final performance metrics will be displayed upon completion of training.