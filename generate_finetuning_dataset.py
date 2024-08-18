import pandas as pd
import random
from nlp_synt_data import *
from data.prompts import *
from data.texts import *
from datasets import Dataset, DatasetDict

if __name__ == '__main__':

    def generate_diff_len(texts:list[tuple[str,str]],):
        return texts


    res_all_dict = {}
    res_all_dict = {
        'train':{"act":[], "conversations":[],"split":[]},
        'test':{"act":[], "conversations":[],"split":[]},
                }

    TEXTS = [
        (TEXT_JOB_v0,'all'),
        (TEXT_JOB_TRAIN_v0,'train'),
        (TEXT_JOB_TEST_v0,'test'),
        (JOBS_OTHER_SPLIT_v0_n0,'other'),
    ]

    for TEXT in TEXTS:

        prompts = PromptGenerator.generate(PROMPTS_JOB_V0, [['zsl',],])
        data = DataGenerator.generate(TEXT[0], SUBS_JOBS_V0)
        try:
            ResponseGenerator.generate(f"data/raw_data_finetune_{TEXT[1]}.csv", data, prompts, lambda prompt, text: "", n_pass=1)
        except:
            print("")

        df = pd.read_csv(f'data/raw_data_finetune_{TEXT[1]}.csv')
        print("reading", len(df), "rows")

        res_dict = {"act":[], "conversations":[],"split":[]}

        for act in [
            # 'LABEL',
            'EXPLAIN',
            ]:
            def generate(row):
                prompt = PromptGenerator.get(row['prompt_id'], PROMPTS_JOB_V0)
                text = DataGenerator.get(row['text_id'], TEXT[0], SUBS_JOBS_V0)
                response = ""
                if text['text'][1] == 'TODO':
                    t = ("","")
                    if 'JOB' in text['keys']:
                        t = text['keys']['JOB']
                    elif 'ADJ' in text['keys']:
                        t = text['keys']['ADJ']
                    response = 'INCLUSIVO' if t[1] == 'neutro' else 'NON INCLUSIVO'
                    if 'EXPLAIN' in act:
                        w = t[0]
                        gend = t[1]
                        response = f'la frase contiene la parola "{w}" che è di genere "{gend}" e quindi la risposta è "{response}"'
                else:
                    response = text['text'][1]
                    if 'EXPLAIN' in act:
                        response = f'la frase si riferisce ad un pubblico generale neutro e la risposta è quindi "{response}"' if response == 'INCLUSIVO' else f'la frase non si riferisce ad un pubblico generale neutro e la risposta è quindi "{response}"'
                return [ { "from": "human", "value": prompt + " " + text['text'][0] }, { "from": "gpt", "value": response } ]
            df['response'] = df.apply(generate, axis=1)
            res = df['response'].tolist()
            res_dict["conversations"] = res_dict["conversations"] + res
            res_dict["act"] = res_dict["act"] + [act]*len(res)
            res_dict["split"] = res_dict["split"] + [TEXT[1]]*len(res)

        # res_all_dict[TEXT[1]] = res_dict
        def add(original, to_add):
            for k in ['conversations','act','split']:
                original[k] = original[k] + to_add[k]

        if TEXT[1] == 'all' or TEXT[1] == 'other':
            _dict = Dataset.from_dict(res_dict).train_test_split(test_size=0.3, seed=42)
            add(res_all_dict['train'], _dict['train'].to_dict())
            add(res_all_dict['test'], _dict['test'].to_dict())
        elif TEXT[1] == 'train' or TEXT[1] == 'test':
            print('hola')
            add(res_all_dict[TEXT[1]], res_dict)


    # random.shuffle(res_dict["conversations"])
    # dataset = Dataset.from_dict(res_all_dict)
    # split_dataset = dataset.train_test_split(test_size=0.3, seed=42)

    dataset_dict = DatasetDict({
        # 'train': dataset,
        'train': Dataset.from_dict(res_all_dict['train']),
        'test': Dataset.from_dict(res_all_dict['test']),
    })

    # dataset_dict = DatasetDict({
    #     'train': all_splt['train'] + other_splt['train'] + train_splt,
    #     'test': all_splt['test'] + other_splt['test'] + test_splt,
    # })

    # dataset_dict = DatasetDict({
    #     # 'train': dataset,
    #     'train': split_dataset['train'],
    #     'test': split_dataset['test']
    # })

    print(dataset_dict)
    print(dataset_dict['train'][0])
    print(dataset_dict['train'][-1])

    # print(dataset_dict['train']['conversations'][:1])
    # dataset_dict.push_to_hub("romabob/unimi-job", token="")
