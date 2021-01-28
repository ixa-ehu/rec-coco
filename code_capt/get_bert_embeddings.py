import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
#import logging
#logging.basicConfig(level=logging.INFO)

#import matplotlib.pyplot as plt
#% matplotlib inline

import csv
import numpy as np
import json

def load_dict_data(fileDir):
    # Opens the visual genome parsed data and gets a dictionary for EACH SAMPLE\                                                             
    DICT_DATA, i = [], 0

    with open(fileDir) as f:
      for line in f:
                i = i + 1
                splits = line.rstrip('\n').split(',')
                if i == 1:
                    var_names = splits
                    var_names = [var_names[j].replace('\r', '') for j in range(len(var_names))]
                else:
                    new_example = {}
                    if len(var_names) == len(splits):
                        for j in range(len(var_names)):
                            new_example[var_names[j]] = splits[j]
                        DICT_DATA.append(new_example)

    return DICT_DATA

def load_training_data(fileDir):
    #INPUT: fileDir: is the filename that the function load_dict_data function above needs                                                                     
    #OUTPUT: a vector (of samples) of each of the columns (variables) of the tr\aining data                                                                     
    dict_data = load_dict_data(fileDir)
    var_names = [key for key in dict_data[0]]
    VECTORS = {}
    for k in range(len(var_names)):
        VECTORS[var_names[k]] = []
    for i in range(len(dict_data)):
        for j in range(len(var_names)):
            VECTORS[var_names[j]].append( dict_data[i][var_names[j]] )
    return VECTORS


training_data = load_training_data('../training_data/Training_data_vic-coco_capt.csv')

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

myfile = open('../embeddings/bert_token_embeddings.json', 'w')
#wr = csv.writer(myfile)


print('Tokenizing Captions...')


captions = training_data['cap']
indexed_captions = []
segments_ids = []
tokenized_captions = []
for caption in captions:
  marked_text = "[CLS] " + caption + " [SEP]"
  tokenized_text = tokenizer.tokenize(marked_text)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  segments_id = [1] * len(tokenized_text)
  tokenized_captions.append(tokenized_text)
  indexed_captions.append(indexed_tokens)
  segments_ids.append(segments_id)

print('Loading BERT model...')
#print(indexed_captions[0])
#print(segments_ids[0])


# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

print('Predicting embeddings...')

torch.cuda.set_device(0)

ids_dict = {}
ids_dict['tokens'] = []
ids_dict['vectors'] = []

for i, indexed_caption in enumerate(indexed_captions):
    token_emb_dict = {}

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_caption]).to('cuda')
    segments_tensors = torch.tensor([segments_ids[i]]).to('cuda')
    model.to('cuda')

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
        token_vecs_sum = []
    # Convert the hidden state embeddings into single token vectors

    # Holds the list of 12 layer embeddings for each token
    # Will have the shape: [# tokens, # layers, # features]
    token_embeddings = [] 
    batch_i = 0
    tokens  = []
    # For each token in the sentence...
    for token_i in range(len(tokenized_captions[i])):
        tokens.append(tokenized_captions[i][token_i])
        
        # Holds 12 layers of hidden states for each token 
        hidden_layers = [] 

        # For each of the 12 layers...
        for layer_i in range(len(encoded_layers)):

            # Lookup the vector for `token_i` in `layer_i`
            vec = encoded_layers[layer_i][batch_i][token_i]

            hidden_layers.append(vec)
            
        token_embeddings.append(hidden_layers)
    
    # For each token in the sentence...
    for token in token_embeddings:
        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(torch.stack(token)[-4:], 0).cpu().numpy().tolist()
    
        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec)
    ids_dict['tokens'].append(tokens)
    ids_dict['vectors'].append(token_vecs_sum)
    
json.dump(ids_dict, myfile)
#wr.writerows(token_vecs_sum)
    
    
myfile.close()
