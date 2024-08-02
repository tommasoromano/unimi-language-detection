import pandas as pd
import random
from nlp_synt_data import *
from data.prompts import *
from data.texts import *
from datasets import Dataset, DatasetDict

if __name__ == '__main__':

    prompts = PromptGenerator.generate(PROMPTS_JOB_V0, PROMPTS_JOB_V0_KEYS_FINETUNING)
    data = DataGenerator.generate(TEXT_JOBS_v0, SUBS_JOBS_V0)

    # {'id': 't#0_JOB#0', 'text': 'Studio Legale Rossi & Associati cerca giornaista Legale', 'text_label': 'TODO', 'info': {'JOB': {'value': 'giornaista', 'label': 'neutro'}}}
    # 

    # ResponseGenerator.generate(f"data/raw_data_finetune.csv", data, prompts, lambda prompt, text: "", n_pass=1)

    df = pd.read_csv('data/raw_data_finetune.csv')

    res_dict = {"act":[], "conversations":[]}
    for act in ['LABEL','EXPLAIN']:
        def generate(row):
            prompt = PromptGenerator.get(row['prompt_id'], PROMPTS_JOB_V0)
            text = DataGenerator.get(row['text_id'], TEXT_JOBS_v0, SUBS_JOBS_V0)
            response = ""
            if text['text'][1] == 'TODO':
                response = 'INCLUSIVO' if text['keys']['JOB'][1] == 'neutro' else 'NON INCLUSIVO'
                if 'EXPLAIN' in act:
                    job = text['keys']['JOB'][0]
                    gend = text['keys']['JOB'][1]
                    response = f'la frase contiene la parola "{job}" che è di genere "{gend}" e quindi la risposta è "{response}"'
            else:
                response = text['text'][1]
                if 'EXPLAIN' in act:
                    response = f'la frase si riferisce ad un pubblico generale neutro e la risposta è quindi "{response}"' if response == 'INCLUSIVO' else f'la frase non si riferisce ad un pubblico generale neutro e la risposta è quindi "{response}"'
            return [ { "from": "human", "value": prompt + " " + text['text'][0] }, { "from": "gpt", "value": response } ]
        df['response'] = df.apply(generate, axis=1)
        res = df['response'].tolist()
        res_dict["conversations"] = res_dict["conversations"] + res
        res_dict["act"] = res_dict["act"] + [act]*len(res)

    random.shuffle(res_dict["conversations"])
    dataset = Dataset.from_dict(res_dict)
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

    dataset_dict = DatasetDict({
        'train': dataset,
        # 'train': split_dataset['train'],
        # 'test': split_dataset['test']
    })

    print(dataset_dict)
    print(dataset_dict['train'][0])
    print(dataset_dict['train']['conversations'][:5])
    # dataset_dict.push_to_hub("romabob/unimi-job", token="")
