import argparse
import os
import re
import json
import sys
import string
import pandas as pd
import numpy as np
import math
from collections import defaultdict, Counter
from functools import partial
import os
import glob
from typing import Dict, List

import torch
import transformers
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding)
from datasets import Dataset, load_dataset

from pprint import pprint
from tqdm import tqdm

device = "cuda:0"# "cuda" if torch.cuda.is_available() else "cpu"

from modules.metrics import *

TOKENIZER = None
INPUT_SEQUENCE_MAX_LENGTH = 512


def init_mt5_base_tokenizer():
    global TOKENIZER
    TOKENIZER = MT5Tokenizer.from_pretrained('google/mt5-base')

def add_eos_to_examples(example):
    
    # compatible with xsquad
    result = {}
    context = example['context']
    question = example['question']

    if 'answers' in example.keys() and type(example['answers']) == dict:
        result['answers'] = example['answers']['text'] 
        answer = result['answers'][0]
    elif 'answers' in example.keys() and type(example['answers']) == list and type(example['answers'][0]) == dict:
        answer = example['answers'][0]['text']
        result['answers'] = example['answers']
    elif 'answers' in example.keys()  and type(example['answers']) == list and type(example['answers'][0]) == str:
        answer = example['answers'][0]
        result['answers'] = example['answers']
  
    else:
        answer = example['answer']

    result['input_text'] =  'question: %s context: %s' % (question, context)
    result['answer'] = result
    result['target_text'] = '%s' % answer
    
    return result


def convert_to_features(example, tokenizer):
    assert tokenizer is not None

    encoding = {}
    input_encoding = tokenizer.encode_plus(example['input_text'],
                                        #    pad_to_max_length=True,
                                           truncation=True,
                                           max_length=INPUT_SEQUENCE_MAX_LENGTH,
                                           add_special_tokens=True)
    target_encoding = tokenizer.encode_plus(example['target_text'],
                                            pad_to_max_length=True,
                                            truncation=True,
                                            max_length=64,
                                            add_special_tokens=True)

    encoding['input_ids'] = input_encoding['input_ids']
    encoding['target_ids'] = target_encoding['input_ids']

    return encoding


def get_squad_answer_str(context, qas):
    context_qa_pairs = []
    for qa in qas:
        qid = qa['id']
        question = qa['question']
        answer = qa['answers'][0]['text']
        answers = list(map(lambda x: x['text'], qa['answers']))
        answer_start = qa['answers'][0]['answer_start']
        context_qa_pairs.append((qid, context, question, answer, answer_start, answers))
    return context_qa_pairs

def process_squad_en(squad_en: Dict[str, List[Dict]]):
    squad_dataset = defaultdict(lambda : dict())
    for split_name in ['train', 'validation']:
        for i, item in enumerate(squad_en[split_name]):
            paragraphs = item['paragraphs']
    #         print('.' ,end='')
            for j, paragraph in enumerate(paragraphs):

                context = paragraph['context']
                context_qa_pairs = get_squad_answer_str(context=context, qas=paragraph['qas'])

                for context_qa_pair in context_qa_pairs:
                    qid, context, question, answer, answer_start, answers = context_qa_pair

                    qa_item = {
                        'qid': qid,
                        'question': question,
                        'context': context,
                        'answer': answer,
                        'answers': answers,
                        'answer_start': answer_start,
                    }
                    squad_dataset[split_name][qid] = qa_item
        
        print(f'Number of {split_name} examples: {len(squad_dataset[split_name]):,}')
    
    return squad_dataset


def main(args):

    init_mt5_base_tokenizer()

    data_dir = args.data_dir
    output_dir = args.output_dir
    test_dataset_type = args.test_dataset_type 


    features, references, references_lang = None, None, None
    xorqa_xx_dir = mlqa_xx_dir = squad_xx_dir = os.path.join(data_dir, 'datasets_format')

    if test_dataset_type == 'squad':
        squad_en_dir = os.path.join(data_dir, 'en')

        squad_en = { 
            'train': json.load(open(os.path.join(squad_en_dir, 'train-v1.1.json')))['data'],
            'validation': json.load(open(os.path.join(squad_en_dir, 'dev-v1.1.json')))['data']
        }
        
        squad_en_processed = process_squad_en(squad_en)
        squad_en_val_processed = list(squad_en_processed['validation'].values())
        _convert_to_features = partial(convert_to_features, tokenizer=TOKENIZER)
        features = list(map(_convert_to_features, map(add_eos_to_examples, squad_en_val_processed)))
        references = [item['answers'] for item in squad_en_val_processed]
    elif test_dataset_type == 'xquad':
        squad_xx = { 
            'test': json.load(open(os.path.join(squad_xx_dir, 'test.json')))['data']
        }
   
        xquad_test_dataset = squad_xx['test']
        _convert_to_features = partial(convert_to_features, tokenizer=TOKENIZER)
        features = list(map(_convert_to_features, map(add_eos_to_examples, xquad_test_dataset)))
        references_lang = list(map(lambda x: x['lang'], xquad_test_dataset))
        references = [ list(map(lambda x: x['text'], item['answers'])) for item in xquad_test_dataset ]
    elif test_dataset_type == 'mlqa':
        mlqa_xx = { 
            'test': json.load(open(os.path.join(mlqa_xx_dir, 'test.json')))['data']
        }
        mlqa_test_dataset = mlqa_xx['test']
        _convert_to_features = partial(convert_to_features, tokenizer=TOKENIZER)
        features = list(map(_convert_to_features, map(add_eos_to_examples, mlqa_test_dataset)))
        references_lang = list(map(lambda x: x['lang'], mlqa_test_dataset))
        references = [ item['answers']['text'] for item in mlqa_test_dataset ]
    elif test_dataset_type == 'xorqa':
        xorqa_xx = { 
            'test': json.load(open(os.path.join(xorqa_xx_dir, 'test.json')))['data']
        }
        xorqa_test_dataset = xorqa_xx['test']
        _convert_to_features = partial(convert_to_features, tokenizer=TOKENIZER)
        features = list(map(_convert_to_features, map(add_eos_to_examples, xorqa_test_dataset)))
        references_lang = list(map(lambda x: x['lang'], xorqa_test_dataset))
        references = [ item['answers']['text'] for item in xorqa_test_dataset ]
    else:
        raise ValueError('The value of `test_dataset_type` should be from the following values [`squad`, `xquad`, `mlqa`, `xorqa`].')

    finetuned_model_dir = args.finetuned_model_dir
    each_ckp_finetuned_model_dirs = glob.glob(os.path.join(finetuned_model_dir, 'checkpoint-*'))
    assert len(each_ckp_finetuned_model_dirs) >= 1
    

    batch_size = args.batch_size
    generation_max_length = args.generation_max_length
    generation_beam_size = args.generation_beam_size

    per_checkpoint_scores = []
    per_lang_per_checkpoint_scores = []
    per_example_scores = []
    per_lang_per_example_scores = []
    
    for each_ckp_finetuned_model_dir in each_ckp_finetuned_model_dirs:

        print('='*50)
        print(f'\n\n\nFinetuned model directory: {each_ckp_finetuned_model_dir}')
        
        model_exp_name = finetuned_model_dir.split('/')[-1]
        model_checkpoint = each_ckp_finetuned_model_dir.split('-')[-1] # pattern: `checkpoint-([\d]+)`
 
        model = MT5ForConditionalGeneration.from_pretrained(os.path.join(finetuned_model_dir, f'checkpoint-{model_checkpoint}')).to('cuda:0')
        model.eval()

        

        data_collator = DataCollatorForSeq2Seq(tokenizer=TOKENIZER,
                                                padding=True,
                                                max_length=INPUT_SEQUENCE_MAX_LENGTH)
        data_loader = torch.utils.data.DataLoader(features,
                                                  batch_size=batch_size,
                                                  collate_fn=data_collator)
        predictions = []
        c = 0
        batched_gt_ans = []
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
 
            target_ans = TOKENIZER.batch_decode(batch['target_ids'], skip_special_tokens=True)
            batched_gt_ans.extend(target_ans)
            
            outs = model.generate(input_ids=batch['input_ids'].to(device), 
                                attention_mask=batch['attention_mask'].to(device),
                                max_length=generation_max_length,
                                early_stopping=True,
                                num_beams=generation_beam_size,
                                decoder_start_token_id=0)

            answer = [TOKENIZER.decode(ids, skip_special_tokens=True) for ids in outs]
            if c < 5:
    
                batch_level_eval_results = evaluate(references[i*batch_size:(i+1)*batch_size], answer[:batch_size])
                print(' - Batch-level eval:')
                pprint(batch_level_eval_results, indent=4)
                zipped_ref_ans = list(zip(references[i*batch_size:(i+1)*batch_size - abs(batch_size - 10)],
                                     answer[:10]))
                print(f'zipped_ref_ans:\n{zipped_ref_ans}\n')
                print('\n')
                c+=1

            predictions.extend(answer)


        per_example_scores.append(per_example_evaluate(references, predictions))
        eval_results = evaluate(references, predictions)           
        
        print('\nPer-epoch eval results:')
        print(eval_results)
        print('\n')
        per_checkpoint_scores.append({
            'model_ckp': model_checkpoint,
            'model_dir': each_ckp_finetuned_model_dir,
            **eval_results
        })

        if references_lang is not None:
            per_lang_per_example_scores.append(per_example_evaluate_with_lang(references, predictions, references_lang))
            per_lang_eval_results = evaluate_with_lang(references, predictions, references_lang)
            print(f'\nDEBUG: per_lang_eval_results: {per_lang_eval_results}\n')
            per_lang_per_checkpoint_scores.append({
            'model_ckp': model_checkpoint,
            'model_dir': each_ckp_finetuned_model_dir,
            **per_lang_eval_results
        })

        print(f'DEBUG: len(per_checkpoint_scores): {len(per_checkpoint_scores)}')
        print(f'\nDEBUG: per_checkpoint_scores')
        pprint(per_checkpoint_scores)
        print('\n')
    target_result_dir = os.path.join(output_dir, test_dataset_type)
    if not os.path.exists(target_result_dir):
        os.makedirs(target_result_dir, exist_ok=True)
    
    target_result_per_checkpoint_path = os.path.join(target_result_dir, f'{test_dataset_type}.{model_exp_name}.per_checkpoint.json')
    target_result_per_example_path = os.path.join(target_result_dir, f'{test_dataset_type}.{model_exp_name}.per_example.json')
  
    with open(target_result_per_checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(per_checkpoint_scores, f, indent=4)
    with open(target_result_per_example_path, 'w', encoding='utf-8') as f:
        json.dump(per_example_scores, f, indent=4)

    if references_lang:
        target_result_per_lang_per_checkpoint_path = os.path.join(target_result_dir, f'{test_dataset_type}.{model_exp_name}.per_checkpoint.per_lang.json')
        target_result_per_lang_per_example_path = os.path.join(target_result_dir, f'{test_dataset_type}.{model_exp_name}.per_example.per_lang.json')

        with open(target_result_per_lang_per_checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(per_lang_per_checkpoint_scores, f, indent=4)
        with open(target_result_per_lang_per_example_path, 'w', encoding='utf-8') as f:
            json.dump(per_lang_per_example_scores, f, indent=4)

    print('\n')
    print('-'*60)
    print('\n\n')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('finetuned_model_dir', type=str)
    parser.add_argument('--data_dir', type=str, default='./data/xquad')
    parser.add_argument('--test_dataset_type', type=str, default='squad')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--input_sequence_max_length', type=int, default=512)
    parser.add_argument('--generation_max_length', type=int, default=64)
    parser.add_argument('--generation_beam_size', type=int, default=1)


    args = parser.parse_args()
    
    INPUT_SEQUENCE_MAX_LENGTH = args.input_sequence_max_length

    main(args)