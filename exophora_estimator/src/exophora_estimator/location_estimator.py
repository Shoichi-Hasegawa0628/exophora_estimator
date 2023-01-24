import numpy as np
import pandas as pd
import openai
import random
import time


class Location_Estimator:
    def __init__(self, obj_plc_matrix, org_id, api_key, gpt_model='text-curie-001'):
        """Main class for object-place estimator using GPT-3.

        Args:
            obj_plc_matrix (str): the path of the csv file which contains the object-place matrix
            org_id (str): Path to the file that contains the organization id
            api_key (str): Path to the file that contains the api key
            gpt_model (str, optional): Pretrained GPT model. One of the following: 'text-ada-001', 'text-babbage-001', 'text-curie-001', 'text-davinci-002'. Defaults to 'text-curie-001'.
        """

        # initialize the openai api
        with open(org_id, 'r') as f:
            org_id = f.read().rstrip("\n")
        with open(api_key, 'r') as f:
            api_key = f.read().rstrip("\n")
        openai.organization = org_id
        openai.api_key = api_key
        openai.Model.list()
        self.gpt_model = gpt_model

        # load the csv file as pandas dataframe with header
        obj_plc_matrix = pd.read_csv(obj_plc_matrix, header=0)
        # extract the column names and raw names and convert them to list
        self.obj_list = obj_plc_matrix.columns.tolist()
        self.plc_list = obj_plc_matrix.index.tolist()
        # replace '_' to ' ' in the obj_list and plc_list
        self.obj_list = [x.replace('_', ' ').lower() for x in self.obj_list]
        self.plc_list = [x.replace('_', ' ').lower() for x in self.plc_list]
        # extract probability matrix from the dataframe
        self.plc_given_obj = obj_plc_matrix.values.T
    
    def sample_plc_given_obj(self, target_obj):
        """Sample a place given an object from the object-place matrix.
        
        Args:
            target_obj (str): The target object.
        
        Returns:
            str: The sampled place."""
        
        # find the index of the object
        obj_index = self.obj_list.index(target_obj)

        # sample the place given the object
        plc = np.random.choice(self.plc_list, p=self.plc_given_obj[obj_index, :])

        return plc
    
    def estimate_location(self, target_obj, num_trial=10, num_obj_sample_per_trial=5, min_total_prob=0.0):
        """Estimate the location of the target object using GPT-3.
        
        Args:
            target_obj (str): The target object to estimate its location.
            num_trial (int, optional): The number of trials. Defaults to 10.
            num_obj_sample_per_trial (int, optional): The number of object samples per trial. Defaults to 5.
            min_total_prob (float, optional): The minimum total probability of the sampled locations. Defaults to 0.0.
        
        Returns:
            dict: Including keys 'estimation' (dict), 'prompts' (list), 'num_total_tokens' (int), 'cost' (float) and 'run_time' (float).
            """
        
        # log the start time
        start = time.time()
        
        # replace '_' to ' ' in the target_obj
        target_obj = target_obj.replace('_', ' ').lower()

        # the dictionary to store the probability of each place
        prob_dict = {}
        # the list to store prompts
        prompt_list = []
        # the number of succeed trials which means the total probability is over the threshold
        num_sampled = 0
        # the number of total tokens used by GPT-3
        num_total_tokens = 0

        # loop over the number of trials
        for trial in range(num_trial):
            # the list to store the sampled object-place pairs
            obj_plc_pairs = []

            # sample object from self.obj_list
            sampled_obj_list = random.sample(self.obj_list, num_obj_sample_per_trial)
            # for each sampled object, sample a place from the object-place matrix
            for obj in sampled_obj_list:
                plc = self.sample_plc_given_obj(obj)
                obj_plc_pairs.append([obj, plc])
            
            # the instruction prompt for GPT-3
            instruction = 'For an object, answer where it is placed in the house.'

            # format the example for GPT-3 from the sampled object-place pairs
            example = []
            for pair in obj_plc_pairs:
                obj = pair[0]
                plc = pair[1]
                example.append(f'Object: {obj}\nLocation: {plc}')
            example = '\n'.join(example)

            # the target prompt for GPT-3
            target_prompt = f'Object: {target_obj}\nLocation:'

            # merge the instruction, example, and target prompt
            prompt = f'{instruction}\n{example}\n{target_prompt}'
            # append the prompt to the prompt_list
            prompt_list.append(prompt)

            # feed the prompt to GPT-3 and get the response
            response = openai.Completion.create(
                                                model=self.gpt_model,
                                                prompt=prompt,
                                                temperature=0,
                                                logprobs=5,
                                                stop='\n'
                                                )
            
            # extract log probabilities from the response
            log_probs = response['choices'][0]['logprobs']['top_logprobs'][0]
            # convert dictionary of log probabilities to dictionary of probabilities
            probs = {k.replace(' ',''): np.exp(v) for k, v in log_probs.items()}

            # check whether total probs over the threshold
            if sum(probs.values()) > min_total_prob:
                num_sampled += 1
                # add the probability of each place to the prob_dict
                for k, v in probs.items():
                    if k in prob_dict:
                        prob_dict[k] += v
                    else:
                        prob_dict[k] = v
            # update the number of total tokens used by GPT-3
            num_total_tokens += response['usage']['total_tokens']
        # devide the probability of each place by the number of sampled trials and normalize the probability
        for k, v in prob_dict.items():
            prob_dict[k] = v/num_sampled
        prob_dict = {k: v/sum(prob_dict.values()) for k, v in prob_dict.items()}
        # sort the probability in descending order
        prob_dict = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

        # calculate the cost for the estimation
        if 'ada' in self.gpt_model:
            cost = num_total_tokens/1000 * 0.0004
        elif 'babbage' in self.gpt_model:
            cost = num_total_tokens/1000 * 0.0005
        elif 'curie' in self.gpt_model:
            cost = num_total_tokens/1000 * 0.002
        elif 'davinci' in self.gpt_model:
            cost = num_total_tokens/1000 * 0.02

        return {'estimation': prob_dict, 'prompts': prompt_list, 'num_total_tokens': num_total_tokens, 'cost': cost, 'run_time': time.time()-start}
