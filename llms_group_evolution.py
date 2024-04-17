#modify a trl demo to orgnize a group of llms to generate more positive comments to a movie, a bert based sentiments classifier acts as reward model

'''
group evolution fake code:
for i in num_evolution
    for j in num_epoch
        for k in num_llms
            train_ppo(llms[k])
            dump positive sample to llms_positive_sample[k]
        for k in num_llms
            train_sft(llms[k],llms_positive_sample[!=k])        
    replace the Bottom N llms with dupliations of the Top M llms
'''
from dataclasses import dataclass, field
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed, SFTTrainer
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available
from typing import Optional
import pandas as pd
import os

base_model_path = "/DATA/jupyter/personal/gpt2"
out_put_path = "/DATA/jupyter/personal/group_evolution"
tuned_model_path = out_put_path + "/models"
dumped_positive_review_path = out_put_path + "/positive_reviews"
reward_model_path = "/DATA/jupyter/personal/lvwerra/distilbert-imdb"
datasets_parquet_path = "/DATA/jupyter/personal/imdb/plain_text"
num_evolution = 1
num_epoch = 2
num_llms = 2
# early break a epoch if converge or for tuning efficiency sake
max_ppo_steps_per_epoch = 1
#generated review by LLM is kept as training data for sft, if scentiment score above the threshhold 
positive_sample_scentiment_threshhold = 1.2

tqdm.pandas()

#make sure the path exist,if !exist then mkdir 
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, 0o777)

@dataclass
class ScriptArguments:
    use_seq2seq: bool = field(default=False, metadata={"help": "whether to use seq2seq"})
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})

    # LoraConfig
    use_peft: bool = field(default=False, metadata={"help": "whether to use peft"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})

parser = HfArgumentParser((ScriptArguments, PPOConfig))
args, ppo_config = parser.parse_args_into_dataclasses()
# modify local base model, local sentiments analiser, peft dataset...
ppo_config.model_name = base_model_path
ppo_config.query_dataset = datasets_parquet_path
ppo_config.reward_model = "sentiment-analysis:" + reward_model_path
ppo_config.mini_batch_size = 128
ppo_config.batch_size = 128

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}
trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead

# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(config, query_dataset, input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        query_dataset (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets.here we load local data in parquet format 
    # ds = load_dataset(query_dataset, split="train")
    ds = load_dataset(
        path = "parquet", 
        data_dir = query_dataset, 
        data_files = {'train': 'train-00000-of-00001.parquet', 'test': 'test-00000-of-00001.parquet'},
        split = "train"
        )

    ds = ds.rename_columns({"text": "review"})
    # only long review is used to train
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)
    # 2~8 head words is used,instant the sampler class
    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(ppo_config, ppo_config.query_dataset)

#confusion solved：遍历字典就是遍历key，遍历值需要遍历 dict.values（），遍历KV：xx.items() here，we trans the list of dicts to a dict of list
def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

# set seed before initializing value head for deterministic eval
set_seed(ppo_config.seed)

#init base model:
# Now let's build the model, the reference model, and the tokenizer.
# why peft donot need ref_model: no need ref model by default,that is what PEFT mean.
if not args.use_peft:
    ref_model = trl_model_class.from_pretrained(ppo_config.model_name, trust_remote_code=args.trust_remote_code)
    device_map = None
    peft_config = None
else:
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
    ref_model = None
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}

model = trl_model_class.from_pretrained(
    ppo_config.model_name,
    trust_remote_code=args.trust_remote_code,
    device_map=device_map,
    peft_config=peft_config,
)

tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
tokenizer.pad_token_id = tokenizer.eos_token_id

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)
# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}

#init a reward model:(bert scentiment classifier) as environment to score a response
# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    if is_xpu_available():
        device = "xpu:0"
    elif is_npu_available():
        device = "npu:0"
    else:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
task, model_name = ppo_config.reward_model.split(":")
if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
    with ds_plugin.zero3_init_context_manager(enable=False):
        sentiment_pipe = pipeline(task, model=model_name, device=device)
else:
    sentiment_pipe = pipeline(task, model=model_name, device=device)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
if sentiment_pipe.tokenizer.pad_token_id is None:
    sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id
if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id

#loop for evolution start
for i in range(num_evolution):
    #loop for epoch start
    for j in range(num_epoch):
        #inner loop for rl start
        for k in range(num_llms):
            #PPO train: here we use a sequnential ppo training to save GPU memory
            print("RL from feedback from envrionment in evol: %d, epoch: %d, llms: %d begin:" % (i, j, k))
            #init the peft model[k] with different adaptor for specific llms,over write ppotrainer
            if i == 0 & j == 0:
                #as inited above,all models inited with the same basemodel/peft model
                print("cold start in evol: %d, epoch: %d, llms: %d with model: %s" % (i, j, k, base_model_path))  
            else:
                #load saved model from last epoch,over write ppotrainer
                model_save_path = "%s/model_aftersft_evolve%d_epoch%d_llms%d" % (tuned_model_path, i, (j-1), k)
                print("hot  start in evol: %d, epoch: %d, llms: %d with model: %s" % (i, j, k, model_save_path))
                ppo_config.model_name = model_save_path
                #hot start ref model
                if not args.use_peft:
                    ref_model = trl_model_class.from_pretrained(ppo_config.model_name, trust_remote_code=args.trust_remote_code)
                    device_map = None
                    peft_config = None
                else:
                    peft_config = LoraConfig(
                        r=args.lora_r,
                        lora_alpha=args.lora_alpha,
                        bias="none",
                        task_type="CAUSAL_LM",
                    )
                    ref_model = None
                    # Copy the model to each device
                    device_map = {"": Accelerator().local_process_index}
                #load hot start model
                model = trl_model_class.from_pretrained(
                    ppo_config.model_name,
                    trust_remote_code=args.trust_remote_code,
                    device_map=device_map,
                    peft_config=peft_config,
                )
                #only over write ppo trainer with hot start,reward model \ tokenizer \ dataset \ devices keep as initialized 
                ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator) 
            #PPO training
            num_batch = 0
            positive_reviews = []
            for _epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
                query_tensors = batch["input_ids"]
                
                #model[k] generate a response ,using a randomly sampled query
                # Get response from gpt2
                response_tensors, ref_response_tensors = ppo_trainer.generate(
                    query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
                )
                batch["response"] = tokenizer.batch_decode(response_tensors)
                batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

                #reward model give the responses a score
                # Compute sentiment score
                texts = [q + r for q, r in zip(batch["query"], batch["response"])]
                pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
                rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
                ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
                ref_pipe_outputs = sentiment_pipe(ref_texts, **sent_kwargs)
                ref_rewards = [torch.tensor(output[1]["score"]) for output in ref_pipe_outputs]
                batch["ref_rewards"] = ref_rewards
                # print batch_avg_rewards before each ppo step
                batch_rewards_avg = sum(rewards) / len(rewards)
                print("\nIn in evol: %d, epoch: %d, llms: %d,Before PPO Step:%d,rewards_avg = %f \n" % (i, j, k, num_batch, batch_rewards_avg))
                #accumulate positive reviews over all batchs from k_th llm
                batch_positive_review = [review for review, reward in zip(texts, rewards) if (reward > positive_sample_scentiment_threshhold)]
                positive_reviews.extend(batch_positive_review)
                # Run PPO step
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])
            
                num_batch += 1
                # ppo can converge before all training datas are consumed,also for tuning efficiency sake.add a ealy_stop switch
                if num_batch == max_ppo_steps_per_epoch:
                    break     
            print("RL from feedback from envrionment in evol: %d, epoch: %d, llms: %d end:" % (i, j, k))
            
            #save model per epoch
            model_save_path = "%s/model_afterppo_evolve%d_epoch%d_llms%d" % (tuned_model_path, i, j, k)
            ensure_dir(model_save_path)
            ppo_trainer.save_pretrained(model_save_path)
            #save generation as positive sample if reward is bigger than threshhold
            df = pd.DataFrame(
                {
                "dumped_positive_generation":positive_reviews    
                }
            )            
            sample_save_path = "%s/generated_positive_reviews_evolve%d_epoch%d_llms%d.parquet" % (dumped_positive_review_path, i, j, k)
            ensure_dir(sample_save_path)
            df.to_parquet(sample_save_path, engine='pyarrow')
            #ppo train next llms
        
        #inner rl loop over all llms end

        #inner sft loop over all llms start
        for k in range(num_llms):
            print("SFT from other llm's expericences in evol: %d, epoch: %d, llms: %d " % (i, j, k))
            #A model sfted by positive sample from all other models, vice versa
            #load positive generations from other llms
            all_positivesample_exceptself = []
            for m in range(num_llms):
                #skip selfgenerated sample
                if m == k:
                    continue
                #load sample from other llms
                sample_save_filename = "generated_positive_reviews_evolve%d_epoch%d_llms%d.parquet" % (i, j, k)
                dataset_sft = load_dataset(
                    path = "parquet", 
                    data_dir = dumped_positive_review_path, 
                    data_files = {'train': sample_save_filename},
                    split = "train"
                    )
                all_positivesample_exceptself.append(dataset_sft)
            #concat datasets
            dataset_sft = concatenate_datasets(all_positivesample_exceptself)
            #load tuned model after ppo
            model_save_path = "%s/model_afterppo_evolve%d_epoch%d_llms%d" % (tuned_model_path, i, j, k)
            model = AutoModelForCausalLM.from_pretrained(model_save_path)
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            # train
            trainer = SFTTrainer(
                model,
                train_dataset=dataset_sft,
                dataset_text_field="dumped_positive_generation",
                max_seq_length=512,
            )
            trainer.train()
            #model save checkpoint
            model_save_path = "%s/model_aftersft_evolve%d_epoch%d_llms%d" % (tuned_model_path, i, j, k)
            trainer.save_model(model_save_path)
            #eval with reward model,the dataset is original dataset with length sample
            dataloader = DataLoader(dataset, batch_size = 128, collate_fn =  collator)
            num_batch = 0
            for batch in dataloader:
                #generation
                review_head = batch["query"]
                generater = pipeline(
                    "text-generation",
                    model = model,
                    tokenizer=tokenizer,
                    device="cuda"
                    )
                generated_reviews = generater(review_head,
                    **generation_kwargs
                    )
                #compute scentiment rewards
                texts = [review[0]["generated_text"] for review in generated_reviews]
                pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
                rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
                batch_rewards_avg = sum(rewards) / len(rewards)
                print("\nIn in evol: %d, epoch: %d, llms: %d,After all SFT batchs,eval with batch data:%d,rewards_avg = %f \n" % (i, j, k, num_batch, batch_rewards_avg))
                num_batch += 1
                # use same amount of batchs to check rewards, as ppo step did
                if num_batch == max_ppo_steps_per_epoch:
                    break
        #inner sft loop end
    #loop for epoch end

#drop the bottom models,duplicate the top models

#loop for evolution end

print("End")
