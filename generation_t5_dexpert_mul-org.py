import torch
from transformers import T5ForConditionalGeneration,T5TokenizerFast
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import GPUtil
import re, regex
import json, sys, regex
import argparse
import logging
import glob
import os
from tqdm import tqdm, trange
import pandas as pd
import torch.nn as nn

from pathlib import Path
from typing import Union, List

import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from generation.gpt2_generation import GPT2Generation

from utils import utils
from utils.generation_utils import top_k_top_p_filtering

global device, device_ids
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
device_ids = GPUtil.getAvailable(limit = 4)

class CustomDataset(Dataset):

    def __init__(self, dataframe, source_cls, target_cls):
        self.data = dataframe
        self.comment_text = self.data.source
        self.org_text = self.data.target

        self.labels = self.data.label

        self.source_cls = source_cls
        self.target_cls = target_cls

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        source_text = str(self.comment_text[index])
        org_tweet =  str(self.org_text[index])

        label = str(self.labels[index])
        
        input_st = source_text #.replace(self.source_cls+"_CLS: ", "") 

        return {
            'para_text': source_text,
            'trans_input_text': input_st,
            'org_tweet': org_tweet,
            'label': label
        }

def regular_encode(args, file_path, shuffle=True, num_workers = 1, batch_size=64, maxlen = 32, mode = 'train'):
    
    # if we are in train mode, we will load two columns (i.e., text and label).
    if mode == 'train':
        # Use pandas to load dataset
        df = pd.read_csv(file_path, delimiter='\t',header=0, names=['source', 'target', 'label'], encoding='utf-8', quotechar='"')

    else:
        print("the type of mode should be either 'train' or 'predict'. ")
        return
        
    print("{} Dataset: {}".format(file_path, df.shape))
    
    custom_set = CustomDataset(df, args.source_cls, args.target_cls)
    
    dataset_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': num_workers}

    batch_data_loader = DataLoader(custom_set, **dataset_params)
    
    return batch_data_loader


MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

class DExpertsGeneration: 
    STOP_TOKEN = "</s>"

    def __init__(
        self, 
        base_model: Union[str, Path, T5ForConditionalGeneration],
        antiexpert_model: Union[str, Path, T5ForConditionalGeneration] = None,
        expert_model: Union[str, Path, T5ForConditionalGeneration] = None,
        tokenizer: str = 't5-base', 
        seed: int = 42,
        expert_prefix: str = None,
        antiexpert_prefix: str = None
        ):
        # Set up device
        self.expert_prefix = expert_prefix
        self.antiexpert_prefix = antiexpert_prefix
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        utils.set_seed(seed, n_gpu)

        self.base_model = T5ForConditionalGeneration.from_pretrained(base_model).to(self.device)
        
        if antiexpert_model:
            self.antiexpert = T5ForConditionalGeneration.from_pretrained(antiexpert_model).to(self.device)
        else:
            self.antiexpert = None
        
        if expert_model:
            self.expert = T5ForConditionalGeneration.from_pretrained(expert_model).to(self.device)
        else:
            self.expert = None
        
        self.tokenizer = T5TokenizerFast.from_pretrained(base_model)
        self.tokenizer.pad_token = self.STOP_TOKEN
        assert self.tokenizer.eos_token_id == self.tokenizer.pad_token_id

    def generate(self,
             para_text: Union[str, List[str]],
             base_source: Union[str, List[str]],
             max_len: int = 20,
             sample: bool = True,
             filter_p: float = 0.9,
             k: int = 0,
             p: float = 1.0,
             temperature: float = 1.0,
             alpha: float = 0.0
            ):
        
        if isinstance(para_text, str):
            para_text = [para_text]
        else:
            para_text = para_text
            
        if isinstance(base_source, str):
            base_source = [base_source]
        else:
            base_source = base_source
            
        source_base = ["paraphrase: " + x for x in base_source]
        
        source_expert = para_text
        source_antiexpert = para_text
        
        target = []
        for x in source_base:
            target.append("<pad>")
            
        encodings_dict_base = self.tokenizer.batch_encode_plus(source_base, pad_to_max_length=True, return_tensors='pt')
        input_ids_base = encodings_dict_base['input_ids'].to(self.device)
        attention_mask_base = encodings_dict_base['attention_mask'].to(self.device)
        
        encodings_dict_expert = self.tokenizer.batch_encode_plus(source_expert, pad_to_max_length=True, return_tensors='pt')
        input_ids_exper = encodings_dict_expert['input_ids'].to(self.device)
        attention_mask_exper = encodings_dict_expert['attention_mask'].to(self.device)
        
        
        encodings_dict_anti = self.tokenizer.batch_encode_plus(source_antiexpert, pad_to_max_length=True, return_tensors='pt')
        input_ids_anti = encodings_dict_anti['input_ids'].to(self.device)
        attention_mask_anti = encodings_dict_anti['attention_mask'].to(self.device)
        
        decoder_dict = self.tokenizer.batch_encode_plus(target, return_tensors='pt')
        decoder_input_ids = decoder_dict['input_ids'].to(self.device)
        decoder_attention_mask = decoder_dict['attention_mask'].to(self.device)
        
        batch_size, input_seq_len = input_ids_exper.shape

        unfinished_sents = torch.ones(batch_size, dtype=torch.long, device=self.device)

        self.base_model.eval()
        if self.expert:
            self.expert.eval()
        if self.antiexpert:
            self.antiexpert.eval()
        with torch.no_grad():
            for step in range(max_len):
                # base model prediction
                #print("base ", source_base[0])
                base_logits = self.base_model(input_ids_base, attention_mask = attention_mask_base, 
                                              decoder_input_ids = decoder_input_ids, decoder_attention_mask = decoder_attention_mask)["logits"]
                
                # expert prediction
                if self.expert:
                    #print("expert ", source_expert[0])
                    expert_logits = self.expert(input_ids_exper, attention_mask = attention_mask_exper, 
                                               decoder_input_ids = decoder_input_ids, decoder_attention_mask = decoder_attention_mask)["logits"]
                else:
                    expert_logits = base_logits
                
                # antiexpert prediction
                if self.antiexpert:
                    #print("antiexpert ", source_antiexpert[0])
                    antiexpert_logits = self.antiexpert(input_ids_anti, attention_mask = attention_mask_anti, 
                                                       decoder_input_ids = decoder_input_ids, decoder_attention_mask = decoder_attention_mask)["logits"]
                else:
                    antiexpert_logits = base_logits
                
                if filter_p < 1.0:
                    base_logits = top_k_top_p_filtering(base_logits, top_p=filter_p)
                
                # DExperts
                alpha = torch.tensor(alpha).to(self.device)
                ensemble_logits = base_logits + alpha * (expert_logits - antiexpert_logits)

                next_token_logits = ensemble_logits[:, -1, :]

                if sample:
                    # Temperature (higher temperature => more likely to sample low probability tokens)
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    if k > 0 or p < 1.0:
                        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=k, top_p=p)
                    # Sample
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                   # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                # either append a padding token here if <EOS> has been seen or append next token
                tokens_to_add = next_tokens * unfinished_sents + self.tokenizer.pad_token_id * (1 - unfinished_sents)
                # this updates which sentences have not seen an EOS token so far
                # if one EOS token was seen the sentence is finished
                eos_in_sents = tokens_to_add == self.tokenizer.eos_token_id
                unfinished_sents.mul_((~eos_in_sents).long())

                # stop when there is an EOS in each sentence
                if unfinished_sents.max() == 0:
                    break

                # Update input_ids, attention_mask and position_ids
                decoder_input_ids = torch.cat([decoder_input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                decoder_attention_mask = torch.cat([decoder_attention_mask, decoder_attention_mask.new_ones((batch_size, 1))], dim=1)

        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                           for output in decoder_input_ids[:, :]]
        return decoded_outputs, source_base, source_expert, source_antiexpert

def tweet_normalizer(txt):
    txt.replace("<unk>","").replace("<unk>","<pad>")
    # remove duplicates
    temp_text = regex.sub("(USER\s+)+","USER ", txt)
    temp_text = regex.sub("(URL\s+)+","URL ", temp_text)
    temp_text = re.sub("[\r\n\f\t]+","",temp_text)
    temp_text = re.sub(r"\s+"," ", temp_text)
    temp_text = regex.sub("(USER\s+)+","USER ", temp_text)
    temp_text = regex.sub("(URL\s+)+","URL ", temp_text)
    
    return temp_text

def main():
    
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

#     parser.add_argument("--model_path", default=None, type=str, required=True,
#                     help="Path to pre-trained model or shortcut name")

    parser.add_argument("--base_model_name", default=None, type=str, required=True,
                    help="Path to base model name")
    
    parser.add_argument("--expert_model_name", default=None, type=str, required=True,
                    help="Path to expert model name")
    
    parser.add_argument("--anti_model_name", default=None, type=str, required=True,
                    help="Path to antiexpert model name")

    parser.add_argument("--source_cls", default=None, type=str, required=True,
                    help="Prefix of source input")

    parser.add_argument("--target_cls", default=None, type=str, required=True,
                    help="Prefix of target style")
    
    parser.add_argument("--orginal_base", action='store_true',
                help="Whether give the orignial text as base model's input.")
    
    ## Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size GPU/CPU.")

    parser.add_argument("--num_workers", default=1, type=int,
                        help="Total number of num_workers.")

    parser.add_argument("--num_return", default=1, type=int,
                        help="Total number of generated paraphrases per tweet.")

    parser.add_argument("--top_k", default=0, type=int,
                        help="Total number of generated paraphrases per tweet.")

    parser.add_argument("--top_p", default=0.95, type=float,
                        help="Total number of generated paraphrases per tweet.")
    
    parser.add_argument("--alpha", default=2.0, type=float,
                    help="Control style transfer")


    args = parser.parse_args()

    
    generator = DExpertsGeneration(
        base_model=args.base_model_name, 
        expert_model=args.expert_model_name,
        antiexpert_model=args.anti_model_name
    )
    
    
    train_file = args.input_file

    train_dataloader = regular_encode(args, train_file, batch_size=args.batch_size)

    file_name = args.input_file.split("/")[-1].replace(".tsv", "")
    
    model_name = "dexpert-mul"
    
    if args.orginal_base:
        model_name = model_name + "-org"
    
    output_file = os.path.join(args.output_dir, "{}-{}_{}-{}_{}_transfer.json".format(file_name, model_name, args.source_cls, args.target_cls,str(args.top_p)))
    
    if os.path.exists(output_file):
        os.remove(output_file)

    for ind, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        
        labels = batch["label"]
        para_texts = batch["para_text"]
        org_tweets = batch["org_tweet"]
        trans_input_text = batch["trans_input_text"]
        
        if args.orginal_base: 
            base_source = org_tweets
        else:
            base_source = trans_input_text
        
        final_outputs, source_base, source_expert, source_antiexpert = generator.generate(
         para_text = trans_input_text,
         base_source = base_source,
         max_len = 128,
         sample = False,
         filter_p = 0.9,
         k = args.top_k,
         p = args.top_p,
         temperature = 1.0,
         alpha = args.alpha
                )
        
        if ind == 0:
            print("source_base", source_base)
            print("source_expert", source_expert)
            print("source_antiexpert", source_antiexpert)
        
        output_lines = []
        for ind in range(len(labels)):
            para_text = para_texts[ind]
            org_tweet = org_tweets[ind]

            paraphrases = final_outputs[ind * args.num_return : (ind+1) * args.num_return]
            paraphrases = [tweet_normalizer(x) for x in paraphrases]
            
            output_all = {}
            output_all["para_text"] = para_text
            output_all["org_tweet"] = org_tweet
            output_all["source_cls"] = args.source_cls
            output_all["target_cls"] = args.target_cls
            
            output_all["label"] = labels[ind]
            
            output_all["paraphrase"] = paraphrases[0]
            
            output_lines.append(json.dumps(output_all)+"\n")
            
        with open(output_file, "a") as out_f:
            out_f.writelines(output_lines)
    

if __name__ == "__main__":
    main()