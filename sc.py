import pandas as pd
from nlp_synt_data import *
from data.prompts import *
from data.texts import *
from analysis_utils import *

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
