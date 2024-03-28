from transformers import AutoTokenizer, pipeline, AutoModel
import torch
import bitsandbytes
import accelerate

'''
group evolution fake code:
for i in num_evolution
    for i in num_epoch
        for j in num_llms
            train_ppo(llms[j])
            dump positive sample to llms_positive_sample[j]
        for j in num_llms
            train_sft(llms[j],llms_positive_sample[!=j])        
    replace the Bottom N llms whith a dupliation of the Top M llms
'''




#init a reward model(bert scentiment classifier) as environment to score a response

#init base model

#loop for evolution start

#loop for epoch start

#inner loop for rl start

#PPO train: here we use a sequnential ppo training to save GPU memory

#init the peft model with different adaptor for specific llms

#model[i] generate a response ,using a randomly sampled query

#reward model give the responses a score

#run a PPO step,and save the generation as positive sample if reward is bigger than threshhold

#model save checkpoint
#evaluate with reward model

#ppo train next llms

#inner rl loop end

#inner sft loop start

#A model sfted by positive sample from all other models, vice versa

#model save checkpoint
#eval with reward model

#inner sft loop end



#loop for epoch end

#drop the bottom models,duplicate the top models

#loop for evolution end

print("end")