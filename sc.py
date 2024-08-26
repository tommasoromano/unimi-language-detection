import pandas as pd
from nlp_synt_data import *
from data.prompts import *
from data.texts import *
from analysis_utils import *

def text_maker(texts:list[tuple[str,str]], n_chars):
    res = []
    new_text = ""
    new_label = "INCLUSIVO"
    for text,label in texts:
        if label == "TODO":
            new_label = label
        new_text += "\n" + text
        if len(new_text) > n_chars:
            res.append((new_text,new_label))
            new_text = ""
            new_label = "INCLUSIVO"
    if new_text != "":
        res.append((new_text,new_label))
        new_text = ""
        new_label = "INCLUSIVO"
    return res

def sc_df_finetuned_seed(remove_reinference=False):
    model ='phi3-finetuned'
    df = pd.read_csv(f'results/{model}-seed-eval-v0.csv')
    df = df[~df['response'].str.contains('NOT IN TEST DATASET')]
    if remove_reinference:
        df = df[df['text_id'].str[1] == "#"]
    df['text_id'] = df.apply(lambda x: x['text_id'] if x['text_id'][1] == '#' else "t" + x['text_id'][2:], axis=1)
    df['text_id'] = df.apply(lambda x: x['text_id'] if x['text_id'][1] == '#' else "t" + x['text_id'][2:], axis=1)
    df['text'] = df.apply(lambda x: DataGenerator.get(x['text_id'],
                                                        TEXT_SEED_v0(),
                                                        SUBS_JOBS_V0,
                                                        )['text'][0],
                        axis=1)
    # df = df[df['prompt_id'].str.contains('cot')]

    df_fix = fix_df(df, model, show_plot=False)
    return df_fix

def sc_make_test_data():
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
    return DataGenerator.generate(texts(), SUBS_JOBS_V0)