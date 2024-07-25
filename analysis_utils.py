import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as skm
LABELS = ['INCLUSIVO', 'NON INCLUSIVO']

def fix_df(df:pd.DataFrame, model):
    df['text_JOB_value'] = df.apply(lambda x: x['text_JOB_value'] if isinstance(x['text_JOB_value'],str) else "", axis=1)

    df['true'] = df.apply(lambda x: ('INCLUSIVO' if x['text_JOB_label'] == 'neutro' else 'NON INCLUSIVO') if x['text_labels'] == 'TODO' else x['text_labels'], axis=1)

    def fix_gemma2(df):
        fdf = df.copy()
        def _f(x):
            r = x['response'].upper()
            if r.startswith('NON INCLUSIVO'):
                return 'NON INCLUSIVO'
            elif r.startswith('INCLUSIVO'):
                return 'INCLUSIVO'
            return None
        fdf['response'] = fdf.apply(lambda x: _f(x), axis=1)
        return fdf

    def fix_mistral(df):
        fdf = df.copy()
        def _f(x):
            r = x['response'].upper()
            r = r[1:]
            if r.startswith('NON INCLUSIVO'):
                return 'NON INCLUSIVO'
            elif r.startswith('INCLUSIVO'):
                return 'INCLUSIVO'
            return None
        fdf['response'] = fdf.apply(lambda x: _f(x), axis=1)
        return fdf
    
    def fix_qwen2(df):
        fdf = df.copy()
        def _f(x):
            r = x['response'].upper()
            if r == 'NON INCLUSIVO':
                return 'NON INCLUSIVO'
            elif r == 'INCLUSIVO':
                return 'INCLUSIVO'
            return None
        fdf['response'] = fdf.apply(lambda x: _f(x), axis=1)
        return fdf

    df_fix = df.copy()
    if model == 'gemma2':
        df_fix = fix_gemma2(df)
    if model == 'mistral':
        df_fix = fix_mistral(df)
    if model == 'qwen2':
        df_fix = fix_qwen2(df)

    df_fix.dropna(subset=['response'], inplace=True)

    # res = pd.DataFrame({'count':[],'fixed':[],'raw%':[],'fixed%':[],'prompt_id':[],'true':[]})
    res = pd.DataFrame({'count':[],'%':[],'df':[],'prompt_id':[],'true':[]})
    for p in df['prompt_id'].unique():
        for t in df['true'].unique():
            # r = len(df[(df['prompt_id'] == p) & (df['true'] == t)])
            # d = len(df_fix[(df_fix['prompt_id'] == p) & (df_fix['true'] == t)])
            # res = pd.concat([res,pd.DataFrame({'raw':[r],'fixed':[d],'raw%':[round(r/r,2)],'fixed%':[round(d/r,2)],'prompt_id':[p],'true':[t]})])
            r = len(df[(df['prompt_id'] == p) & (df['true'] == t)])
            d = len(df_fix[(df_fix['prompt_id'] == p) & (df_fix['true'] == t)])
            res = pd.concat([res,pd.DataFrame({'count':[r],'%':[round(r/r,2)],'df':'raw','prompt_id':[p],'true':[t]})])
            res = pd.concat([res,pd.DataFrame({'count':[d],'%':[round(d/r,2)],'df':'fixed','prompt_id':[p],'true':[t]})])

    print(res)
    sns.barplot(data=res, x='prompt_id', y='%', hue='df')
    plt.show()
    
    return df_fix

NEUTRALS = [
'giornaista',
'autista',
'dentista',
'ginnasta',
'consulente',
'estetista',
'farmacista',
'contabile',
'nutrizionista',
'giardiniere',
'insegnante',
'igienista dentale',]

def isin(df, ls:list[str], invert=False):
    _df = df.copy()
    if invert:
        _df = _df[~_df['text_JOB_value'].isin(ls)]
    else:
        _df = _df[_df['text_JOB_value'].isin(ls)]
    return _df

def contains(df, ls:list[str], invert=False):
    _df = df.copy()
    for s in ls:
        if invert:
            _df = _df[~_df['text_JOB_value'].str.contains(s)]
        else:
            _df = _df[_df['text_JOB_value'].str.contains(s)]
    return _df

def metrics(df):
    total = len(df)
    true_positives = len(df[(df['true'] == 'INCLUSIVO') & (df['response'] == 'INCLUSIVO')])
    true_negatives = len(df[(df['true'] == 'NON INCLUSIVO') & (df['response'] == 'NON INCLUSIVO')])
    false_positives = len(df[(df['true'] == 'NON INCLUSIVO') & (df['response'] == 'INCLUSIVO')])
    false_negatives = len(df[(df['true'] == 'INCLUSIVO') & (df['response'] == 'NON INCLUSIVO')])
    # print(f"TP: {len(true_positives)}")
    # print(f"TN: {len(true_negatives)}")
    # print(f"FP: {len(false_positives)}")
    # print(f"FN: {len(false_negatives)}")
    try:
        sensitivity = true_positives / (true_positives + false_negatives)
    except:
        sensitivity = None
    try:
        specificity = true_negatives / (true_negatives + false_positives)
    except:
        specificity = None
    try:
        accuracy = (true_positives + true_negatives) / total
    except:
        accuracy = None
    try:
        precision = true_positives / (true_positives + false_positives)
    except:
        precision = None
    try:
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    except:
        f1 = None
    try:
        negative_predictive_value = true_negatives / (true_negatives + false_negatives)
    except:
        negative_predictive_value = None
    # print(f"Sensitivity: {sensitivity}")
    # print(f"Specificity: {specificity}")
    # print(f"Accuracy: {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"F1: {f1}")
    # print(f"Negative Predictive Value: {negative_predictive_value}")
    return {
        'total': total,
        'true_positives': true_positives/total,
        'true_negatives': true_negatives/total,
        'false_positives': false_positives/total,
        'false_negatives': false_negatives/total,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'precision': precision,
        'f1': f1,
        'negative_predictive_value': negative_predictive_value
    }