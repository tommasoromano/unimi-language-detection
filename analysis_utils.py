import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as skm
LABELS = ['INCLUSIVO', 'NON INCLUSIVO']

def fix_df(df:pd.DataFrame, model, show_plot=False):
    df['response_text'] = df.apply(lambda x: x['response'], axis=1)
    if 'text_JOB_value' in df.columns:
        df['text_JOB_value'] = df.apply(lambda x: x['text_JOB_value'] if isinstance(x['text_JOB_value'],str) else "", axis=1)
        df['text_ADJ_value'] = df.apply(lambda x: x['text_ADJ_value'] if isinstance(x['text_ADJ_value'],str) else "", axis=1)
        if 'text_VERB_value' in df.columns: df['text_VERB_value'] = df.apply(lambda x: x['text_VERB_value'] if isinstance(x['text_VERB_value'],str) else "", axis=1)

    def make_label(x):
        if x['text_labels'] != 'TODO':
            return x['text_labels']
        if x['text_JOB_value'] != "":
            gend = x['text_JOB_label']
        elif x['text_ADJ_value'] != "":
            gend = x['text_ADJ_label']
        elif x['text_VERB_value'] != "":
            gend = x['text_VERB_label']
        else:
            gend = 'neutro'
        return 'INCLUSIVO' if gend == 'neutro' else 'NON INCLUSIVO'
    
    df['true'] = df.apply(lambda x: make_label(x), axis=1)


    def fix_cot(x):
        r:str = x['response'].upper()
        if r.endswith('NON INCLUSIVO'):
            return 'NON INCLUSIVO'
        elif r.endswith('INCLUSIVO'):
            return 'INCLUSIVO'
        return None
    
    def fix_gemma2(df):
        fdf = df.copy()
        def _f(x):
            if 'cot' in x['prompt_id']:
                return fix_cot(x)
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
            if 'cot' in x['prompt_id']:
                return fix_cot(x)
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
            if 'cot' in x['prompt_id']:
                return fix_cot(x)
            r = x['response'].upper()
            if r == 'NON INCLUSIVO':
                return 'NON INCLUSIVO'
            elif r == 'INCLUSIVO':
                return 'INCLUSIVO'
            return None
        fdf['response'] = fdf.apply(lambda x: _f(x), axis=1)
        return fdf
    
    def fix_phi3_finetuned(df):
        fdf = df.copy()
        def _f(x):
            r = x['response'].upper()
            if 'NON INCLUSIVO' in r:
                return 'NON INCLUSIVO'
            elif 'INCLUSIVO' in r:
                return 'INCLUSIVO'
            return None
        fdf['response'] = fdf.apply(lambda x: _f(x), axis=1)
        return fdf

    df_fix = df.copy()
    if model == 'gemma2':
        df_fix = fix_gemma2(df)
    elif model == 'mistral':
        df_fix = fix_mistral(df)
    elif model == 'qwen2':
        df_fix = fix_qwen2(df)
    elif model == 'phi3-finetuned':
        df_fix = fix_phi3_finetuned(df)
    elif model == 'phi3':
        df_fix = fix_phi3_finetuned(df)
    else:
        raise ValueError(f"Model {model} not supported")

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

    if show_plot:
        print(res)
        sns.barplot(data=res, x='prompt_id', y='%', hue='df')
        plt.title(f"Raw vs Fixed - {model}")
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
    gt_pos = max(1,len(df[df['true'] == 'INCLUSIVO']))
    gt_neg = max(1,len(df[df['true'] == 'NON INCLUSIVO']))
    pred_pos = len(df[df['response'] == 'INCLUSIVO'])
    pred_neg = len(df[df['response'] == 'NON INCLUSIVO'])
    true_positives = len(df[(df['true'] == 'INCLUSIVO') & (df['response'] == 'INCLUSIVO')])
    true_negatives = len(df[(df['true'] == 'NON INCLUSIVO') & (df['response'] == 'NON INCLUSIVO')])
    false_positives = len(df[(df['true'] == 'NON INCLUSIVO') & (df['response'] == 'INCLUSIVO')])
    false_negatives = len(df[(df['true'] == 'INCLUSIVO') & (df['response'] == 'NON INCLUSIVO')])

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
        
    return {
        'total': total,
        'gt_pos': gt_pos,
        'gt_neg': gt_neg,
        'pred_pos': pred_pos,
        'pred_neg': pred_neg,
        'gt_pos%': gt_pos/total,
        'gt_neg%': gt_neg/total,
        'pred_pos%': pred_pos/total,
        'pred_neg%': pred_neg/total,
        'pred/gt_pos%': (pred_pos)/gt_pos,
        'pred/gt_neg%': (pred_neg)/gt_neg,
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

def plot_metrics(res_df, title):
    res_df_melt = res_df.melt(id_vars=['name'], var_name='metric', value_name='value')
    return
    sns.barplot(data=res_df_melt[res_df_melt['metric'].isin([
        'pred/gt_pos%',
        'pred/gt_neg%',
    ])], x='metric', y='value', hue='name')
    plt.title(f"Groud Truth vs Pred{title}")
    plt.show()

    sns.barplot(data=res_df_melt[res_df_melt['metric'].isin([
        'gt_pos',
        'gt_neg',
        'pred_pos',
        'pred_neg',
    ])], x='metric', y='value', hue='name')
    plt.title(f"Groud Truth vs Pred{title}")
    plt.show()

    sns.barplot(data=res_df_melt[res_df_melt['metric'].isin([
        'gt_pos%',
        'gt_neg%',
        'pred_pos%',
        'pred_neg%',
    ])], x='metric', y='value', hue='name')
    plt.title(f"Groud Truth vs Pred{title}")
    plt.show()

    sns.barplot(data=res_df_melt[res_df_melt['metric'].isin([
        'gt_pos%',
        'gt_neg%',
        'pred_pos%',
        'pred_neg%',
    ])], x='name', y='value', hue='metric')
    plt.title(f"Groud Truth vs Pred{title}")
    plt.show()

    sns.barplot(data=res_df_melt[res_df_melt['metric'].isin([
        'true_positives',
        'true_negatives',
        'false_positives',
        'false_negatives',
    ])], x='metric', y='value', hue='name')
    plt.title(f"Confusion Matrix{title}")
    plt.show()

    sns.barplot(data=res_df_melt[res_df_melt['metric'].isin([
        'true_positives',
        'true_negatives',
        'false_positives',
        'false_negatives',
    ])], x='name', y='value', hue='metric')
    plt.title(f"Confusion Matrix{title}")
    plt.show()

    ax = sns.barplot(data=res_df_melt[res_df_melt['metric'].isin([
        'sensitivity',
        'specificity',
        'accuracy',
        'precision',
        'f1',
        # 'negative_predictive_value',
    ])], x='metric', y='value', hue='name')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title(f"Metrics{title}")
    plt.show()

    ax = sns.barplot(data=res_df_melt[res_df_melt['metric'].isin([
        'sensitivity',
        'specificity',
        'accuracy',
        'precision',
        'f1',
        # 'negative_predictive_value',
    ])], x='name', y='value', hue='metric')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title(f"Metrics{title}")
    plt.show()

def split_by_len(df:pd.DataFrame, groups=10):
    df['len'] = df.apply(lambda x: len(x['text']), axis=1)
    mn = df['len'].min()
    mx = df['len'].max()
    step = (mx - mn) // groups
    df['len_label'] = df.apply(lambda x: f"{((x['len']+step)//step)*step}", axis=1)
    return df

def confusion_matrix(df):
    cm = skm.confusion_matrix(df['true'], df['response'], labels=LABELS, normalize='true')
    disp = skm.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot()
    plt.show()

def analyze_len(df):
    res_df = None
    for ln in df['len_label'].unique():
    #for ln in df['len'].unique():
        _metrics = metrics(df[df['len_label'] == ln])
        #_metrics = metrics(df_len[df_len['len'] == ln])
        _metrics['name'] = ln
        if res_df is None:
            res_df = pd.DataFrame(_metrics, index=[0])
        else:
            res_df = pd.concat([res_df, pd.DataFrame(_metrics, index=[0])])

    res_df.dropna()
    res_df['name'] = res_df.apply(lambda x: int(x['name']), axis=1)
    res_df = res_df.sort_values('name', ascending=False)
    plt.plot(res_df['name'],res_df['precision'])
    plt.show()

def metrics_of_dfs(dfs:tuple[pd.DataFrame,str]):
    all = metrics(dfs[0][0])
    all['name'] = dfs[0][1]
    res_df = pd.DataFrame(all, index=[0])

    for df, nm in dfs[1:]:
        if len(df) == 0:
            continue
        _metrics = metrics(df)
        _metrics['name'] = nm
        res_df = pd.concat([res_df, pd.DataFrame(_metrics, index=[0])])

    return res_df