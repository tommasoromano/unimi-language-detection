import pandas as pd
import random
from nlp_synt_data import *
from data.prompts import *
from data.texts import *
from sc import *
from datasets import Dataset, DatasetDict

if __name__ == '__main__':

    # for the train dataset generate conversations
    # for the test, just the data
    res_all_dict = {}

    # TRAIN
    res_all_dict["train"] = {"act":[], "conversations":[]}

    prompts = PromptGenerator.generate(PROMPTS_JOB_V0, [['zsl',],])
    def texts():
        res = TEXT_JOB_TRAIN_v0.copy()
        for ls in [
            JOBS_SPLIT_v0_n0,
            JOBS_SPLIT_v0_n1,
            JOBS_SPLIT_v0_n2,
            JOBS_SPLIT_v0_n4,
            JOBS_SEED_SPLIT_v0_n0,
            JOBS_SEED_SPLIT_v0_n2,
            JOBS_SEED_SPLIT_v0_n3,
            JOBS_SEED_SPLIT_v0_n4,
            # ADJS_SPLIT_v0_n0,
            # JOBS_OTHER_SPLIT_v0_n0,
        ]:
            res += text_maker(ls, 400)
            res += text_maker(ls, 800)
            res += text_maker(ls, 16000000)
        return res
    data = DataGenerator.generate(texts(), SUBS_JOBS_V0)
    def make_chat(prompt, data:Data):
        response = data.text_label
        if data.text_label == "TODO":
            w = ""
            gend = ""
            res = ""
            if "JOB" in data.info:
                w = data.info["JOB"]["value"]
                gend = data.info["JOB"]["label"]
                res = "INCLUSIVO" if gend == "neutro" else "NON INCLUSIVO"
            response = f'la frase contiene la parola "{w}" che è di genere "{gend}" e quindi la risposta è "{res}"'
        else:
            response = f'la frase si riferisce ad un pubblico generale neutro e la risposta è quindi "{response}"' if response == 'INCLUSIVO' else f'la frase non si riferisce ad un pubblico generale neutro e la risposta è quindi "{response}"'
        return [ 
            { "from": "human", "value": prompt + "\n" + data.text }, 
            { "from": "gpt", "value": response } 
            ]
    res_all_dict["train"]["conversations"] = [
        make_chat(p, d)
        for _,p in prompts
        for d in data
    ]
    res_all_dict["train"]["act"] = ["EXPLAIN"]*len(res_all_dict["train"]["conversations"])

    # TEST
    res_all_dict["test"] = {'act':[],'conversations':[]}

    def texts():
        res = TEXT_JOB_TEST_v0.copy()
        for ls in [
            JOBS_SPLIT_v0_n3,
            JOBS_SEED_SPLIT_v0_n1,
        ]:
            res += text_maker(ls, 400)
            res += text_maker(ls, 800)
            res += text_maker(ls, 16000000)
        return res
    data = DataGenerator.generate(texts(), SUBS_JOBS_V0)
    
    res_all_dict["test"]["conversations"] = [
        d.text
        for _,p in prompts
        for d in data
    ]
    res_all_dict["test"]["act"] = ["EXPLAIN"]*len(res_all_dict["test"]["conversations"])

    print(sc_make_test_data()[0])
    exit()

    # publish
    dataset_dict = DatasetDict({
        # 'train': dataset,
        'train': Dataset.from_dict(res_all_dict['train']),
        # 'test': Dataset.from_dict(res_all_dict['test']),
    })

    print(dataset_dict)
    print(dataset_dict['train'][0])
    print(dataset_dict['train'][-1])

    dataset_dict.push_to_hub("romabob/unimi-job", token="")
