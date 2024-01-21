#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import math
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import (
    DataCollatorWithPadding,
    default_data_collator,
    AutoConfig,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForDocumentQuestionAnswering,
    get_scheduler
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from utils_qa import postprocess_qa_predictions
from bbox import get_bbox_from_char_index, normalize_bbox
from ast import literal_eval
from tqdm.auto import tqdm
import os

check_min_version("4.26.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/requirements.txt")

config = {
    'seed': 42,
    'max_seq_length': 512,
    'doc_stride': 128,
    'preprocessing_num_workers': 4,
    'overwrite_cache': False,
    'per_device_train_batch_size': 8,
    'per_device_eval_batch_size': 8,
    'n_best_size': 10,
    'null_score_diff_threshold': 0.0,
    'gradient_accumulation_steps': 1,
    'num_train_epochs': 1,
}

def load_model(architecture, path):
    autoConfig = AutoConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

    if architecture == 'roberta':
        model = AutoModelForQuestionAnswering.from_pretrained(
            path,
            from_tf=False,
            config=autoConfig
        )
    elif architecture == 'impira':
        model = AutoModelForDocumentQuestionAnswering.from_pretrained(
            path,
            from_tf=False,
            config=autoConfig
        )
    return model, tokenizer

def train(architecture, model, tokenizer, dataset, save_path):
    # Create accelerator
    accelerator = Accelerator(gradient_accumulation_steps=config['gradient_accumulation_steps'])

    # Set seed
    if config['seed'] is not None:
        set_seed(config['seed'])

    accelerator.wait_for_everyone()

    column_names = dataset.column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    if architecture == 'impira':
        bbox_column_name = "bbox"
        page_column_name = "page_size"
        config['pad_to_max_length'] = True
    else:
        config['pad_to_max_length'] = False

    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(config['max_seq_length'], tokenizer.model_max_length)

    def prepare_train_features(examples):
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=config['doc_stride'],
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if config['pad_to_max_length'] else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        if architecture == 'impira':
            tokenized_examples["bbox"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]

            if architecture == 'impira':
                context = examples[context_column_name][sample_index]
                bboxes = examples[bbox_column_name][sample_index]
                bboxes = literal_eval(bboxes)
                page_size = examples[page_column_name][sample_index]

                example_bbox = []
                for ind, token in enumerate(offsets):
                    if token == (0, 0) or sequence_ids[ind] != (1 if pad_on_right else 0):
                        example_bbox.append([0, 0, 0, 0])
                    else:
                        token_bbox = list(get_bbox_from_char_index(context, token[0], bboxes))
                        token_bbox = normalize_bbox(token_bbox, page_size[0], page_size[1])
                        example_bbox.append(token_bbox)
                tokenized_examples["bbox"].append(example_bbox)

            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    with accelerator.main_process_first():
        train_dataset = dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=config['preprocessing_num_workers'],
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
    )
        
    if config['pad_to_max_length']:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=config['per_device_train_batch_size']
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=5e-5)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config['gradient_accumulation_steps'])
    max_train_steps = config['num_train_epochs'] * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps * config['gradient_accumulation_steps'],
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config['gradient_accumulation_steps'])
    max_train_steps = config['num_train_epochs'] * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = config['per_device_train_batch_size'] * accelerator.num_processes * config['gradient_accumulation_steps']

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {num_train_epochs}")
    print(f"  Instantaneous batch size per device = {config['per_device_train_batch_size']}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {config['gradient_accumulation_steps']}")
    print(f"  Total optimization steps = {max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= max_train_steps:
                break
    
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    os.makedirs(save_path, exist_ok=True)
    unwrapped_model.save_pretrained(
        save_path, is_main_process=accelerator.is_main_process, save_function=accelerator.save
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(save_path)

def predict(architecture, model, tokenizer, dataset):
    accelerator = Accelerator()
    if config['seed'] is not None:
        set_seed(config['seed'])
    column_names = dataset.column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]

    if architecture == 'impira':
        bbox_column_name = "bbox"
        page_column_name = "page_size"
        config['pad_to_max_length'] = True
    else:
        config['pad_to_max_length'] = False

    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(config['max_seq_length'], tokenizer.model_max_length)

    def prepare_predict_features(examples):
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation = "only_second" if pad_on_right else "only_first",
            max_length = max_seq_length,
            stride = config['doc_stride'],
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if config['pad_to_max_length'] else False,
        )
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []

        if architecture == 'impira':
            tokenized_examples["bbox"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

            if architecture == 'impira':
                context = examples[context_column_name][sample_index]
                bboxes = examples[bbox_column_name][sample_index]
                bboxes = literal_eval(bboxes)
                page_size = examples[page_column_name][sample_index]
                example_bbox = []
                for ind, token in enumerate(tokenized_examples["offset_mapping"][i]):
                    if token == None or sequence_ids[ind] != context_index:
                        example_bbox.append([0, 0, 0, 0])
                    else:
                        token_bbox = list(get_bbox_from_char_index(context, token[0], bboxes))
                        token_bbox = normalize_bbox(token_bbox, page_size[0], page_size[1])
                        example_bbox.append(token_bbox)
                tokenized_examples["bbox"].append(example_bbox)
        return tokenized_examples
    
    with accelerator.main_process_first():
        predict_dataset = dataset.map(
        prepare_predict_features,
        batched=True,
            num_proc=config['preprocessing_num_workers'],
            remove_columns=column_names,
            load_from_cache_file=not config['overwrite_cache'],
            desc="Running tokenizer on prediction dataset",
        )
    if config['pad_to_max_length']:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))
    predict_dataset_for_model = predict_dataset.remove_columns(["example_id", "offset_mapping"])
    predict_dataloader = DataLoader(
        predict_dataset_for_model, collate_fn=data_collator, batch_size=config['per_device_eval_batch_size']
    )

    def post_processing_function(examples, features, predictions, stage="eval"):
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=True,
            n_best_size=config['n_best_size'],
            max_answer_length=30,
            null_score_diff_threshold=config['null_score_diff_threshold'],
            output_dir=None,
            prefix=stage,
        )
        return predictions
    
    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        step = 0
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        for i, output_logit in enumerate(start_or_end_logits):
            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]
            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]
            step += batch_size
        return logits_concat
    
    model, predict_dataloader = accelerator.prepare(
        model, predict_dataloader
    )
    all_start_logits = []
    all_end_logits = []
    model.eval()
    for step, batch in enumerate(predict_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            if not config['pad_to_max_length']:
                start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)
            all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())
    max_len = max([x.shape[1] for x in all_start_logits])
    start_logits_concat = create_and_fill_np_array(all_start_logits, predict_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, predict_dataset, max_len)
    del all_start_logits
    del all_end_logits
    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(dataset, predict_dataset, outputs_numpy)
    return prediction

