
import argparse
from typing import Callable, Dict, List
from simpletransformers.question_answering import QuestionAnsweringModel
import json


DEFAULT_TRAINING_ARGS_MAPPING: Dict[str, Callable] = {
    'xlmroberta': {
        'xlm-roberta-base': lambda output_dir: {
            # 'learning_rate': 3e-5,
            # 'num_train_epochs': 2,
            'max_seq_length': 384,
            'doc_stride': 128,
            'overwrite_output_dir': True,
            'reprocess_input_data': False,
            # 'train_batch_size': 3,
            # 'gradient_accumulation_steps': 8,
            'output_dir': f'{output_dir}',
            'save_eval_checkpoints': False,
            'evaluate_during_training': True,
            'evaluate_during_training_steps': 512,
            'best_model_dir': f"{output_dir}/best_model",
            'evaluate_during_training_verbose': False,
        },
        'xlm-roberta-large': lambda output_dir: {
            # 'learning_rate': 1e-5,
            # 'num_train_epochs': 3,
            'max_seq_length': 512,
            'doc_stride': 128,
            'overwrite_output_dir': True,
            'reprocess_input_data': False,
            # 'train_batch_size': 8,
            # 'gradient_accumulation_steps': 4,
            'output_dir': f'{output_dir}',
            'save_eval_checkpoints': False,
            'evaluate_during_training': True,
            'evaluate_during_training_steps': 512,
            'best_model_dir': f"{output_dir}/best_model",
            'evaluate_during_training_verbose': True,
            'save_optimizer_and_scheduler': False,
            'manual_seed': 2022,
            'warmup_ratio': 0.2,
            'weight_decay': 0.01
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
    
    training_args = DEFAULT_TRAINING_ARGS_MAPPING[model_name][model_variant](output_dir)
    training_args['learning_rate'] = args.learning_rate
    training_args['num_train_epochs'] = args.num_train_epochs
    training_args['warmup_ratio'] = args.warmup_ratio
    training_args['weight_decay'] = args.weight_decay
    training_args['train_batch_size'] = args.train_batch_size
    training_args['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    training_args['wandb_project'] = args.wandb_project
    training_args['wandb_kwargs'] = {
        'name': args.wandb_name 
    }

    print(f'INFO: training_args: {training_args}')

    print('\nINFO: Initialize Trainer')
    
    trainer = QuestionAnsweringModel(model_name, model_variant, training_args)

    print('\nINFO: Begin training')
    trainer.train_model(train_dataset, eval_data=val_dataset)

    print('INFO: Done model finetuning')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_path', type=str)
    parser.add_argument('--val_data_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./checkpoints/squad_extractive.xlm-roberta-base')
    parser.add_argument('--model_name', type=str, default='xlmroberta')
    parser.add_argument('--model_variant', type=str, default='xlm-roberta-base')
    parser.add_argument('--learning_rate', type=float, default=1.5e-5)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--warmup_ratio', type=int, default=0.06)
    parser.add_argument('--weight_decay', type=int, default=0.01)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=6)
    parser.add_argument('--wandb_project', type=str, default='scads.cl-reklt.mrc-training')
    parser.add_argument('--wandb_name', type=str, default='mrc._extractive_reader')

    args = parser.parse_args()

    main(args)

