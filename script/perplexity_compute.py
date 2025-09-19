

import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from tqdm import tqdm

#########
# Parameters
###########

chunksize = 5 # size of each window
stepsize = 5 # how many tokens we move forward at a step
model_name="gpt2-medium"

## Function

# iterate through the text by chunksize/step size,
# compute nll for each chunk
# return all the nlls as a list, and also the mean of them
def likelihood_of_text(text,model,tokenizer,chunksize,stepsize):
    nlls = []
    input_text=tokenizer.encode(text, return_tensors="pt")
    maxsize=input_text.shape[1]
    print('total words '+str(maxsize))
    nsteps=int(np.floor((maxsize-chunksize)/stepsize))
    for i in tqdm(range(nsteps)):
        end_loc = min(i*stepsize+chunksize,maxsize)
        begin_loc = max(0,end_loc-max_length)
        trg_len = chunksize
        input_ids=input_text[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 #don't include loss for all tokens except for the current evaluated sentence
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)
    ppl = (torch.stack(nlls).mean())
    return nlls, ppl

############
# compute
##############

paragraph = "This is a sample text. We want to compute perplexity to it."

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
max_length = model.config.n_positions

nlls,ppl=likelihood_of_text(paragraph,model,tokenizer,chunksize,stepsize)
