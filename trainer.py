import math
import logging
import os
from tqdm import tqdm
import sys
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import torch
from torch.optim import Adam
from datasets import Dataset
from transformers import (
    Trainer,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    Seq2SeqTrainingArguments,
)
from transformers.optimization import get_scheduler
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer
from arguments import ModelArguments, DataTrainingArguments

logger = logging.getLogger(__name__)

class CustomPPOTrainer(PPOTrainer, Trainer):
    r"""
    Inherits PPOTrainer.
    """

    def __init__(
        self,
        model_args: "ModelArguments", 
        data_args: "DataTrainingArguments", 
        training_args: "Seq2SeqTrainingArguments",
        model: "AutoModelForSeq2SeqLMWithValueHead",
        reward_model: Optional["AutoModelForSeq2SeqLMWithValueHead"],
        ref_model: Optional["AutoModelForSeq2SeqLMWithValueHead"],
        tokenizer: "PreTrainedTokenizer",
        data_collator: "DataCollatorWithPadding",
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
    ) -> None:
        backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        ppo_config = PPOConfig(
            model_name=model_args.model_name_or_path,
            mini_batch_size= training_args.per_device_train_batch_size,
            batch_size = backward_batch_size,
            learning_rate=training_args.learning_rate, # 1.41e-5
            # hui add
            accelerator_kwargs={"step_scheduler_with_optimizer": False},
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            optimize_device_cache=True,
            max_grad_norm=1.0,
            whiten_rewards=model_args.whiten_rewards,
            use_score_scaling=model_args.use_score_scaling, # hui scale
            use_score_norm=model_args.use_score_norm, # hui norm
            log_with="wandb",
            seed=42,
            remove_unused_columns = False,
        )
        print('ppo config', ppo_config)

        # Create optimizer and scheduler
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = backward_batch_size 
            num_training_steps = training_args.num_train_epochs * math.ceil(
                len(train_dataset) / total_train_batch_size
            )

        optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=ppo_config.learning_rate,
        )

        scheduler = get_scheduler(
                training_args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=training_args.lr_scheduler_kwargs,
            )
    
        PPOTrainer.__init__(
            self,
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            data_collator=data_collator,
            optimizer=optimizer,
            lr_scheduler=scheduler,
        )
        print('self optimizer', self.optimizer)
        self.args = training_args
        self.model_args = model_args
        self.data_args = data_args
        self.ppo_config = ppo_config
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model

        self.backward_batch_size = backward_batch_size

        self.generation_config = GenerationConfig(
            do_sample=True,
            top_p=1.0,
            top_k=20,
            min_length=5,
            max_length=30,
            temperature=0.9,
            num_return_sequences=1,
            bos_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            
        )

        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        r"""
        Implements training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer.
        """
        if resume_from_checkpoint is not None:
            raise ValueError("`resume_from_checkpoint` will be supported in the future version.")
        
        total_train_batch_size = self.backward_batch_size
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        
        logger.info("***** Running training *****")
        logger.info("  Num examples = {:,}".format(num_examples))
        logger.info("  Num Epochs = {:,}".format(num_train_epochs))
        logger.info("  Instantaneous batch size per device = {:,}".format(self.args.per_device_train_batch_size))
        logger.info(
            "  Total train batch size (w. parallel, buffer, distributed & accumulation) = {:,}".format(
                total_train_batch_size
            )
        )
        logger.info("  Gradient Accumulation steps = {:,}".format(self.args.gradient_accumulation_steps))
        logger.info("  Num optimization epochs per batch = {:,}".format(self.ppo_config.ppo_epochs))
        logger.info("  Total training steps = {:,}".format(max_steps))
        logger.info("  Number of trainable parameters = {:,}".format(
            sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        
        dataiter = iter(self.dataloader)

        for step in tqdm(range(max_steps)):
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)

            # Get inputs
            self.model.eval()
            queries, responses, rewards = [], [], []
            for idx in range(0, self.config.batch_size, self.config.mini_batch_size):
                mini_batch_queries, mini_batch_responses, \
                emo_labels, exp_labels, int_labels = self.my_get_inputs(
                    batch[idx : idx + self.config.mini_batch_size]
                )
                mini_batch_rewards = self.my_get_rewards(mini_batch_queries, mini_batch_responses,
                                                         emo_labels, exp_labels, int_labels)
                queries.extend(mini_batch_queries)
                responses.extend(mini_batch_responses)
                rewards.extend(mini_batch_rewards)

            # Run PPO step
            self.model.train()
            stats = self.step(queries, responses, rewards)

            if self.config.log_with is not None:
                try:
                    batch["query"] = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
                    batch["response"] = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
                    self.log_stats(stats, batch, rewards)
                except Exception:
                    logger.warning("Failed to save stats due to unknown errors.")
            
            self.state.global_step += 1
            if (step + 1) % self.args.save_steps == 0:  # save checkpoint
                self._save_pretrained(
                    os.path.join(self.args.output_dir, "{}-{}".format(PREFIX_CHECKPOINT_DIR, self.state.global_step))
                )

    @torch.no_grad()
    def my_get_inputs(self, batch: Dict[str, "torch.Tensor"]) -> Tuple[List["torch.Tensor"], List["torch.Tensor"]]:
        r"""
        Generates model's responses given queries.
        """
        batch_gen = {'input_ids': batch['input_ids'], 
                     'attention_mask': batch['attention_mask']
                    }
        generate_output = self.model.generate(
            generation_config=self.generation_config, 
            **batch_gen
        )
            
        query = batch["input_ids"].detach().cpu()
        response = generate_output.detach().cpu()

        queries, responses = [], []
        for i in range(len(query)):
            query_length = (query[i] != self.tokenizer.pad_token_id).nonzero().size(0)
            response_indexes = (response[i] != self.tokenizer.pad_token_id).nonzero()
            if len(response_indexes) == 0: # allow empty response
                response_length = 1
            elif self.tokenizer.eos_token_id == self.tokenizer.pad_token_id: # include eos token
                response_length = response_indexes[-1].item() + 2
            else:
                response_length = response_indexes[-1].item() + 1
            queries.append(query[i, :query_length])
            responses.append(response[i, :response_length])

        emo_labels = batch["emotional_reactions"]
        exp_labels = batch["explorations"]
        int_labels = batch["interpretations"]
        batch.pop("emotional_reactions", None)
        batch.pop("explorations", None)
        batch.pop("interpretations", None)
        
        return queries, responses, emo_labels, exp_labels, int_labels

    @torch.no_grad()
    def my_get_rewards(
        self,
        queries: List["torch.Tensor"],
        responses: List["torch.Tensor"],
        emo_labels,
        exp_labels,
        int_labels,
    ) -> List["torch.Tensor"]:
        r"""
        Computes scores using given reward model.

        Both inputs and outputs are put on CPU.
        """
        tokens_queries = \
        self.tokenizer.batch_decode(queries,
                                    skip_special_tokens=True, 
                                    clean_up_tokenization_spaces=True)
        tokens_responses = \
        self.tokenizer.batch_decode(responses,
                                    skip_special_tokens=True, 
                                    clean_up_tokenization_spaces=True)

        rewards = self.reward_model.forward(tokens_queries, tokens_responses, 
                    emo_labels, exp_labels, int_labels)
        
        return rewards.float().detach()  # use fp32 type


