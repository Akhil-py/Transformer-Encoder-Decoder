import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import torch.optim as optim
import matplotlib.pyplot as plt
from transformer import Encoder, Classifier, EncoderWithClassifier, Decoder


from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from utilities import Utilities


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer 1e-3
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        output, loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity


def train_and_evaluate(model, train_loader, test_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9) ###
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9) ###
    
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        total_correct = 0
        total_samples = 0
        
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) ###
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        train_accuracy = 100 * total_correct / total_samples
        train_accuracies.append(train_accuracy)
        
        test_accuracy = evaluate(model, test_loader, device)
        test_accuracies.append(test_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Accuracy: {train_accuracy:.2f}%")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        
        scheduler.step() ###
    
    return train_accuracies, test_accuracies


def evaluate(model, data_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = 100 * total_correct / total_samples
    return accuracy

def visualize_attention(attention_matrix):
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_matrix, cmap='viridis')
    plt.colorbar()
    plt.title('Attention Matrix')
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.show()
    
def train_decoder(decoder, train_loader, optimizer, criterion, max_iters, eval_interval):
    decoder.train()
    total_loss = 0
    for i, (xb, yb) in enumerate(train_loader):
        if i >= max_iters:
            break
            
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
            
        outputs = decoder(xb)
        loss = criterion(outputs.view(-1, outputs.size(-1)), yb.view(-1))
            
        loss.backward()
        optimizer.step()
            
        total_loss += loss.item()
            
        if (i+1) % eval_interval == 0:
            print(f'Epoch {i+1}, Loss: {total_loss / eval_interval}')
            total_loss = 0
            perplexity = compute_perplexity(decoder, train_loader)
            print(f'Training set perplexity after {i+1} Epochs: {perplexity}')
    

def main():    
    parser = argparse.ArgumentParser(description="Run Transformer Parts")
    parser.add_argument("-part1", action="store_true", help="Run only the encoder")
    parser.add_argument("-part2", action="store_true", help="Run only the decoder")
    args = parser.parse_args()
    noArg = not args.part1 and not args.part2
    
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts))
    print("Vocabulary size is", tokenizer.vocab_size)
    
    '''Encoder
    '''
    if args.part1 or noArg:

        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch)

        # Create encoder model
        encoder = Encoder(tokenizer.vocab_size, n_embd, n_head, n_layer, block_size).to(device)
        classifier = Classifier(n_input, n_hidden, n_output).to(device)
        model = EncoderWithClassifier(encoder, classifier).to(device)
        
        # Train and evaluate
        train_accuracies, test_accuracies = train_and_evaluate(model, train_CLS_loader, test_CLS_loader, epochs_CLS, device)
        
        print(f"Final Train Accuracy: {train_accuracies[-1]:.2f}%")
        print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
        
        # Heat map
        sample_input, _ = next(iter(test_CLS_loader))
        sample_input = sample_input.to(device)
        _, attentions = model(sample_input)
        
        # Attention: first head of the last layer
        attention_matrix = attentions[-1][0, 0].cpu().detach().numpy()
        visualize_attention(attention_matrix)
        
        # parameters
        total_params = sum(p.numel() for p in model.encoder.parameters())
        print(f"Number of parameters in the encoder: {total_params}")
        
    '''Decoder
    '''
    if args.part2 or noArg:    
    
        with open('speechesdataset/train_LM.txt', 'r', encoding="utf-8") as f:
            trainLMText = f.read()
        
        train_LM_dataset = LanguageModelingDataset(tokenizer, trainLMText, block_size)
        #print("block size: ", block_size)
        print("train lm dataset len: ", len(train_LM_dataset))
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
        
        # Initialize decoder
        decoder = Decoder(tokenizer.vocab_size, n_embd, n_head, n_layer, n_hidden, block_size, return_attentions=False).to(device)
        optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Train decoder
        train_decoder(decoder, train_LM_loader, optimizer, criterion, max_iters, eval_interval)
        
        # Perplexity
        perplexity = compute_perplexity(decoder, train_LM_loader)
        print(f'Training set perplexity: {perplexity}')

        # Evaluate on test files
        for test_file in ['speechesdataset/test_LM_obama.txt', 'speechesdataset/test_LM_wbush.txt', 'speechesdataset/test_LM_hbush.txt']:
            with open(test_file, 'r', encoding="utf-8") as f:
                text = f.read()
            test_dataset = LanguageModelingDataset(tokenizer, text, block_size)
            #print("test dataset: ", len(test_dataset))
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            perplexity = compute_perplexity(decoder, test_loader)
            print(f'Perplexity on {test_file}: {perplexity}')

        # parameters
        num_params = sum(p.numel() for p in decoder.parameters())
        print(f'Number of parameters in the decoder: {num_params}')


if __name__ == "__main__":
    main()
