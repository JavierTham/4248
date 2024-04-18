# libraries
import nltk
from nltk import RegexpTokenizer
import pandas as pd
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import csv

# loading data
train_1_data = pd.read_csv('data/esnli_train_1.csv')
train_2_data = pd.read_csv('data/esnli_train_2.csv')
test_data = pd.read_csv('data/esnli_test.csv')
combined_train_data = pd.concat([train_1_data, train_2_data], ignore_index=True)
premise, hypothesis, labels = combined_train_data['Sentence1'], combined_train_data['Sentence2'], combined_train_data['gold_label']
premise2, hypothesis2, labels2= test_data['Sentence1'], test_data['Sentence2'], test_data['gold_label']

# Pre-processing using NLTK w/o NER
punc = ['!','.',';','"', '#', '$', '%', '&', '(', ')', '*', "'",'+', ',','=', '-', '--', '/','\\',':', '<', '>', '?', '@', '[', ']', '^', '_', '`', '{', '|', '}', '~']

def preprocess(text):
    toks = []
    words = RegexpTokenizer(r'\w+\'\w{1,2}|\w+').tokenize(str(text)) # skip all punctuations except underscore & word tokenise
    for each in words:
        if each not in punc:
            lowered = each.lower() 
            toks.append(lowered)
    res = " ".join(toks)
    return res

'''
#training
updated_premise = []
updated_hyp = []

for each in premise:
    updated_premise.append(preprocess(each))

for each2 in hypothesis:
    updated_hyp.append(preprocess(each2))
'''

#testing
updated_premise2 = []
updated_hyp2 = []

for each3 in premise2:
    updated_premise2.append(preprocess(each3))

for each4 in hypothesis2:
    updated_hyp2.append(preprocess(each4))

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
if torch.cuda.is_available():
    model = model.to('cuda')
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

def prompt_gen(p,h,l):
    if l=='entailment':
        prompt = f"Explain why {p} has to be true when {h} is true?"
    elif l=='contradiction':
        prompt = f"Explain why {p} cannot be true when {h} is true and vice versa?"
    else: #neutral
        prompt = f"Explain why there is no evidence that if {p} is related to {h}?"
    return prompt

def text_gen(p,h,lab):
  prompt = prompt_gen(p,h,lab)
  input_ids = tokenizer.encode(prompt, return_tensors='pt', 
                               add_special_tokens=False)
  attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
  # Move input data to GPU if available
  if torch.cuda.is_available():
      input_ids = input_ids.to('cuda')
      attention_mask = attention_mask.to('cuda')

  output = model.generate(input_ids,
                          attention_mask=attention_mask,
                          max_length=90,
                          temperature=0.7,
                          top_k=50, 
                          top_p=0.95, 
                          no_repeat_ngram_size=2,
                          num_return_sequences=1)
  return output,prompt

'''
gen_explain_train = []
for ind in range(len(labels)):
    p = updated_premise[ind]
    h = updated_hyp[ind]
    l = labels[ind]

    explanation,prom = text_gen(p,h,l)
    # Decode the generated explanation
    text = tokenizer.decode(explanation[0], skip_special_tokens=True)
    gen_explain_train.append(text[len(prom):].replace('\n','').replace('....',''))

data_train = zip(updated_premise, updated_hyp, gen_explain_train, labels)

# Write the data to a CSV file (TRAIN)
with open('output_gen_train.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the first row with the first elements from each list
    writer.writerow(['premise', 'hypothesis', 'machine_explanation','label'])
    
    # Write the remaining rows
    for row in data_train:
        writer.writerow(row)
'''

#test
gen_explain_test = []
for ind2 in range(3000, 4000):
    p2 = updated_premise2[ind2]
    h2 = updated_hyp2[ind2]
    l2 = labels2[ind2]

    explanation,prom = text_gen(p2,h2,l2)
    # Decode the generated explanation
    text = tokenizer.decode(explanation[0], skip_special_tokens=True)
    gen_explain_test.append(text[len(prom):].replace('\n','').replace('....',''))
    print(f'done generating for {ind2}')

data_test = zip(updated_premise2, updated_hyp2, gen_explain_test, labels2)

# Write the data to a CSV file (TEST)
with open('output_gen_test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the first row with the first elements from each list
    writer.writerow(['premise', 'hypothesis', 'machine_explanation','label'])
    
    # Write the remaining rows
    for row in data_test:
        writer.writerow(row)
