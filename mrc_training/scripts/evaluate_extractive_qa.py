import argparse
from linecache import cache
import os
import re
import json
import sys
import string
import pandas as pd
import numpy as np
import math
from collections import defaultdict, Counter, OrderedDict
from functools import partial
import os
import glob
from typing import Dict, List

import torch
import transformers
from transformers import (
   AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    XLMRobertaTokenizer,
    pipeline,
    default_data_collator
)
from datasets import Dataset, load_dataset

from pprint import pprint
from tqdm import tqdm

DEVICE = "cuda:0" if torch.cuda.is_available() else 'cpu' # "cuda" if torch.cuda.is_available() else "cpu"

from modules.metrics import *

TOKENIZER = None
INPUT_SEQUENCE_MAX_LENGTH = 512

def init_tokenizer(pretrained_tokenizer_name_or_path):
    global TOKENIZER
    
    TOKENIZER = AutoTokenizer.from_pretrained(pretrained_tokenizer_name_or_path,
                                    use_fast=True)
 
def prepare_validation_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    question_column_name = "question" 
    context_column_name = "context"
    answer_column_name = "answer"
    pad_on_right = True
    
    examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    # breakpoint()
    tokenized_examples = TOKENIZER(
        examples[question_column_name],
        examples[context_column_name],
        truncation="only_second",
        max_length=512,
        # return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )
    
    tokenized_examples['target_ids'] = TOKENIZER(
        examples[answer_column_name],
        max_length=64,    
        padding=False,
    )['input_ids']


    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    # sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    # tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        # sample_index = sample_mapping[i]
        # tokenized_examples["example_id"].append(examples["qid"][i])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

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
    squad_dataset = defaultdict(lambda : defaultdict(lambda : list()))
    for split_name in ['train', 'validation']:
        for i, item in enumerate(squad_en[split_name]):
            paragraphs = item['paragraphs']
    #         print('.' ,end='')
            for j, paragraph in enumerate(paragraphs):

                context = paragraph['context']
                context_qa_pairs = get_squad_answer_str(context=context, qas=paragraph['qas'])

                for context_qa_pair in context_qa_pairs:
                    qid, context, question, answer, answer_start, answers = context_qa_pair
 
                    squad_dataset[split_name]['qid'].append(qid)
                    squad_dataset[split_name]['question'].append(question)
                    squad_dataset[split_name]['context'].append(context)
                    squad_dataset[split_name]['answer'].append(answer)
                    squad_dataset[split_name]['answers'].append(answers)
                    squad_dataset[split_name]['answer_start'].append(answer_start)
        
        print(f'Number of {split_name} examples: {len(squad_dataset[split_name]["qid"]):,}')
    
    return squad_dataset

def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    """
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    Args:
        start_or_end_logits(:obj:`tensor`):
            This is the output predictions of the model. We can only enter either start or end logits.
        eval_dataset: Evaluation dataset
        max_len(:obj:`int`):
            The maximum length of the output tensor. ( See the model.eval() part for more details )
    """

    step = 0
    # create a numpy array and fill it with -100.
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
    # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
    for i, output_logit in enumerate(start_or_end_logits):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step

        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat

def _postprocess_qa_predictions(examples,
                               features, 
                               raw_predictions,
                               tokenizer,
                               n_best_size = 20, 
                               max_answer_length = 30,
                               allow_no_answer=False,
                               question_id_col='question_id'):
    
    #get start_logits and end_logits
    all_start_logits, all_end_logits = raw_predictions

    #get `offset_mapping` and `example_id` back
    features.set_format(type=features.format["type"], columns=list(features.features.keys()))

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples[question_id_col])}
    features_per_example = defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["qid"]]].append(i)

    # The dictionaries we have to fill.
    predictions = OrderedDict()
    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]
        context = example["context"]

        min_null_score = None
        valid_answers = []
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            # This is what will allow us to map some the positions in our logits to span of texts in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]

            #debug
            input_ids = features[feature_index]['input_ids'] 

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index-1] is None #end_index is the exclusive upperbound
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index-1][1] #end_index is the exclusive upperbound
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text_decode": tokenizer.decode(input_ids[start_index:end_index]),
                            "text": context[start_char:end_char],
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid failure.
            best_answer = {"text": "", "score": 0.0}
            
        if not allow_no_answer:
            predictions[example[question_id_col]] = best_answer["text"]
        else:
            answer = best_answer["text"]
            predictions[example[question_id_col]] = answer

    return predictions

def main(args):

    init_tokenizer(args.finetuned_model_dir)

    data_dir = args.data_dir
    output_dir = args.output_dir
    test_dataset_type = args.test_dataset_type 


    features, contexts, references, references_lang = None, None, None, None
    xorqa_xx_dir = mlqa_xx_dir = squad_xx_dir = os.path.join(data_dir, 'datasets_format')

    if test_dataset_type == 'squad':
        squad_en_dir = os.path.join(data_dir, 'en')

        squad_en = { 
            'train': json.load(open(os.path.join(squad_en_dir, 'train-v1.1.json')))['data'],
            'validation': json.load(open(os.path.join(squad_en_dir, 'dev-v1.1.json')))['data']
        }
        
        squad_en_processed = process_squad_en(squad_en)
        # squad_en_val_processed = list(squad_en_processed['validation'].values())
        # breakpoint()
        squad_en_val_dataset = Dataset.from_dict(dict(squad_en_processed['validation']))
        # squad_en_val_dataset = squad_en_val_dataset.select(range(1024))
        # squad_en_processed = load_dataset(
        #     'json',
        #     data_files={ 'validation': os.path.join(squad_en_dir, 'dev-v1.1.json') },
        #     field="data",
        #     # cache_dir=model_args.cache_dir,
        #     # use_auth_token=True if model_args.use_auth_token else None,
        # )
        # # _convert_to_features = partial(convert_to_features, tokenizer=TOKENIZER, args=args)
        # features = list(map(_convert_to_features, squad_en_val_processed))
        # contexts = [item['context'] for item in squad_en_val_processed]
        references = [item['answers'] for item in squad_en_val_dataset]
        # column_names = squad_en_val_dataset.column_names
        # eval_dataset = squad_en_val_dataset.map(
        #     prepare_validation_features,
        #     batched=True,
        #     num_proc=1,
        #     remove_columns=column_names,
        #     desc="Running tokenizer on validation dataset",
        #     load_from_cache_file=True,
        #     cache_file_name='./cached/squad.en.dev.v1.1'

        # )

        # print(f'DEBUG: features: {eval_dataset[0:2]}')
        # print(f'DEBUG: references: {references[0:2]}')
    elif test_dataset_type == 'xquad':
        squad_xx = { 
            'test': json.load(open(os.path.join(squad_xx_dir, 'test.json')))['data']
        }
   
        xquad_test_dataset = squad_xx['test']
        _convert_to_features = partial(convert_to_features, tokenizer=TOKENIZER, args=args)
        features = list(map(_convert_to_features,xquad_test_dataset))
        contexts = [item['context'] for item in xquad_test_dataset]
        references_lang = list(map(lambda x: x['lang'], xquad_test_dataset))
        references = [ list(map(lambda x: x['text'], item['answers'])) for item in xquad_test_dataset ]
    elif test_dataset_type == 'mlqa':
        mlqa_xx = { 
            'test': json.load(open(os.path.join(mlqa_xx_dir, 'test.json')))['data']
        }
        mlqa_test_dataset = mlqa_xx['test']
        _convert_to_features = partial(convert_to_features, tokenizer=TOKENIZER, args=args)
        features = list(map(_convert_to_features, mlqa_test_dataset))
        contexts = [item['context'] for item in mlqa_test_dataset]
        references_lang = list(map(lambda x: x['lang'], mlqa_test_dataset))
        references = [ item['answers']['text'] for item in mlqa_test_dataset ]
    elif test_dataset_type == 'xorqa':
        xorqa_xx = { 
            'test': json.load(open(os.path.join(xorqa_xx_dir, 'test.json')))['data']
        }
        xorqa_test_dataset = xorqa_xx['test']
        _convert_to_features = partial(convert_to_features, tokenizer=TOKENIZER)
        features = list(map(_convert_to_features, xorqa_test_dataset))
        contexts = [item['context'] for item in xorqa_test_dataset]
        references_lang = list(map(lambda x: x['lang'], xorqa_test_dataset))
        references = [ item['answers']['text'] for item in xorqa_test_dataset ]
    else:
        raise ValueError('The value of `test_dataset_type` should be from the following values [`squad`, `xquad`, `mlqa`, `xorqa`].')

    finetuned_model_dir = args.finetuned_model_dir
    each_ckp_finetuned_model_dirs = glob.glob(os.path.join(finetuned_model_dir, 'checkpoint-*'))

    print(f'each_ckp_finetuned_model_dirs: {each_ckp_finetuned_model_dirs}')
    assert len(each_ckp_finetuned_model_dirs) >= 1
    

    batch_size = args.batch_size
    # generation_max_length = args.generation_max_length
    # generation_beam_size = args.generation_beam_size

    per_checkpoint_scores = []
    per_lang_per_checkpoint_scores = []
    per_example_scores = []
    per_lang_per_example_scores = []
    
    for each_ckp_finetuned_model_dir in each_ckp_finetuned_model_dirs:

        print('='*50)
        print(f'\n\n\nFinetuned model directory: {each_ckp_finetuned_model_dir}')
        
        model_exp_name = finetuned_model_dir.split('/')[-1]
        model_checkpoint = each_ckp_finetuned_model_dir.split('/')[-1].replace('checkpoint-', '')
        print(f"DEBUG: model_checkpoint: {model_checkpoint}")
 
        model = AutoModelForQuestionAnswering.from_pretrained(each_ckp_finetuned_model_dir).to(DEVICE)
        model.eval()

        qa_pipeline = pipeline('question-answering', model = model, tokenizer = TOKENIZER)
        predictions = []


        for i in tqdm(range(0, len(squad_en_val_dataset) // batch_size)):
            
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            question = squad_en_val_dataset['question'][start_idx:end_idx]
            context = squad_en_val_dataset['context'][start_idx:end_idx]
            # breakpoint()
            prediction = qa_pipeline(question=question,
                                     context=context,
                                     max_answer_len=args.max_answer_length,
                                     max_seq_len=args.input_sequence_max_length)
            prediction_ans = list(map(lambda x: x['answer'].strip(), prediction))
            predictions.extend(prediction_ans)
   
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
    parser.add_argument('--max_answer_length', type=int, default=64)
 
    args = parser.parse_args()
    
    INPUT_SEQUENCE_MAX_LENGTH = args.input_sequence_max_length

    main(args)