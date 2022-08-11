
import argparse
from typing import Callable, Dict, List
from simpletransformers.question_answering import QuestionAnsweringModel
import json


DEFAULT_TRAINING_ARGS_MAPPING: Dict[str, Callable] = {
    'xlmroberta': {
        'xlm-roberta-base': lambda output_dir: {
            'learning_rate': 3e-5,
            'num_train_epochs': 2,
            'max_seq_length': 384,
            'doc_stride': 128,
            'overwrite_output_dir': True,
            'reprocess_input_data': False,
            'train_batch_size': 3,
            'gradient_accumulation_steps': 8,
            'output_dir': f'{output_dir}',
            'evaluate_during_training':True,
            'evaluate_during_training_steps':512,
            'best_model_dir': f"{output_dir}/best_model",
            'evaluate_during_training_verbose': False,
        }
    }
}

def main(args):

    # Load data
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    output_dir = args.output_dir
    model_name = args.model_name
    model_variant = args.model_variant

     
    train_dataset = json.load(open(train_data_path))
    train_dataset = [item for topic in train_dataset['data'] for item in topic['paragraphs'] ]

    val_dataset = None
    if val_data_path is not None:
        val_dataset = json.load(open(val_data_path))

    val_dataset = [item for topic in val_dataset['data'] for item in topic['paragraphs'] ]  
    
    training_args = DEFAULT_TRAINING_ARGS_MAPPING[model_name][model_variant]

    print('INFO: Initialize Trainer')
    
    trainer = QuestionAnsweringModel(model_name, model_variant, training_args)

    print('INFO: Begin training')
    trainer.train_model(train_dataset, eval_data=val_dataset)

    print('INFO: Done model finetuning')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_path', type=str)
    parser.add_argument('--val_data_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./checkpoints/squad_extractive.xlm-roberta-base')
    parser.add_argument('--model_name', type=str, default='xlmroberta')
    parser.add_argument('--model_variant', type=str, default='xlm-roberta-base')

    args = parser.parse_args()

    main(args)

