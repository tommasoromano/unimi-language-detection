import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as skm
from nlp_synt_data import *
from sc import *
from data.texts import *
LABELS = ['INCLUSIVO', 'NON INCLUSIVO']

def fix_df(df:pd.DataFrame):
    df['response_original'] = df.apply(lambda x: x['response'], axis=1)
    df['response'] = df.apply(lambda x: x['response'].upper() if isinstance(x['response'],str) else "", axis=1)

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
    df['len'] = df.apply(lambda x: len(x['text'].split(' ')), axis=1)

    return df

def fix_df_model_response(df:pd.DataFrame, model, show_plot=False):

    def fix(x):
        r = x['response']
        if r.startswith('NON INCLUS'):
            return 'NON INCLUSIVO'
        elif r.startswith('INCLUS'):
            return 'INCLUSIVO'
        if r.startswith(' NON INCLUS'):
            return 'NON INCLUSIVO'
        elif r.startswith(' INCLUS'):
            return 'INCLUSIVO'
        if '"NON INCLUSIVO"' in r:
            return 'NON INCLUSIVO'
        elif '"INCLUSIVO"' in r:
            return 'INCLUSIVO'
        if 'NON È INCLUS' in r:
            return 'NON INCLUSIVO'
        elif 'È NON INCLUS' in r:
            return 'NON INCLUSIVO'
        elif 'È INCLUS' in r:
            return 'INCLUSIVO'
        if "ENTRAMBI" in r:
            return 'INCLUSIVO'
        return None

    df_fix = df.copy()
    # f = lambda x: x['response']
    # if model == 'gemma2':
    #     f = fix_gemma2
    # elif model == 'mistral':
    #     f = fix_mistral
    # elif model == 'llama3':
    #     f = fix_llama
    # elif model == 'qwen2':
    #     df_fix = fix_qwen2(df)
    # elif model == 'phi3-finetuned':
    #     df_fix = fix_phi3_finetuned(df)
    # elif model == 'phi3':
    #     df_fix = fix_phi3_finetuned(df)
    # else:
    #     raise ValueError(f"Model {model} not supported")
    
    df_fix['response'] = df.apply(lambda x: fix(x), axis=1)
    df_fix.dropna(subset=['response'], inplace=True)

    print(f"Fixed {round((len(df)-len(df_fix))/len(df),2)}, {len(df) - len(df_fix)} rows")

    if show_plot:

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
        recall = true_positives / (true_positives + false_negatives)
    except:
        recall = None
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
        f1 = 2 * (precision * recall) / (precision + recall)
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
        'true_positives': true_positives/gt_pos,
        'true_negatives': true_negatives/gt_neg,
        'false_positives': false_positives/gt_neg,
        'false_negatives': false_negatives/gt_pos,
        'recall': recall,
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
        'recall',
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
        'recall',
        'specificity',
        'accuracy',
        'precision',
        'f1',
        # 'negative_predictive_value',
    ])], x='name', y='value', hue='metric')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title(f"Metrics{title}")
    plt.show()

def make_df_len(df:pd.DataFrame, groups=10):
    # df['len'] = df.apply(lambda x: len(x['text']), axis=1)
    mn = 0 # df['len'].min()
    mx = 300 # df['len'].max()
    step = (mx - mn) // groups
    df['len_group'] = df.apply(lambda x: f"{((x['len']+step)//step)*step}", axis=1)
    return df

def make_df_multi_metrics(df_metrics, metrics=[]):
    df_metrics = df_metrics.copy()
    if len(metrics) != 0:
        df_metrics = df_metrics[["name"] + metrics]
    res_df_melt = df_metrics.melt(id_vars=['name'], var_name='metric', value_name='value')
    return res_df_melt

def confusion_matrix(df, true_col='true', pred_col='response'):
    cm = skm.confusion_matrix(df[true_col], df[pred_col], labels=LABELS, normalize='true')
    disp = skm.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot()
    plt.show()

def plot_compare_matrix(df_metrics):
    df = make_df_multi_metrics(df_metrics, [
        'true_positives',
        'true_negatives',
        'false_positives',
        'false_negatives',
    ])
    sns.barplot(data=df, x='metric', y='value', hue='name')
    plt.title(f"Confusion Matrix Comparison")
    plt.show()

def plot_compare_performance_models(df_metrics):
    df = make_df_multi_metrics(df_metrics, [
        'precision',
        'recall',
        'f1',
    ])
    sns.barplot(data=df, x='metric', y='value', hue='name')
    plt.title(f"Performance of Models")
    plt.show()

def fix_df_len_metrics(df_metrics):
    df_metrics = df_metrics.copy()
    df_metrics = df_metrics[df_metrics["name"].str.contains("_")]
    df_metrics["len_group"] = df_metrics.apply(lambda x: int(x['name'].split("_")[1]), axis=1)
    df_metrics["model"] = df_metrics.apply(lambda x: x['name'].split("_")[0], axis=1)
    return df_metrics

def plot_len_metrics(df_metrics, metric_col):
    sns.relplot(
        data=df_metrics[["model","len_group",metric_col]], kind="line",
        x="len_group", y=metric_col, hue="model", markers=True, dashes=False,
    )
    plt.show()

def plot_len_groups(dfs):
    res_df = dfs[0][0].copy()
    res_df["model"] = dfs[0][1]

    for df, nm in dfs[1:]:
        if len(df) == 0:
            continue
        _df = df.copy()
        _df["model"] = nm
        res_df = pd.concat([res_df, _df])
    
    res_df.reset_index(drop=True, inplace=True)

    sns.countplot(data=res_df, x='len_group', hue="model")
    plt.show()

def plot_multi_metrics(df_metrics, metrics=[]):
    df_metrics = make_df_multi_metrics(df_metrics, metrics)
    sns.barplot(data=df_metrics, x='value', y='name', hue='metric')
    plt.show()


def metrics_of_dfs(dfs:list[tuple[pd.DataFrame,str]]):
    all = metrics(dfs[0][0])
    all['name'] = dfs[0][1]
    res_df = pd.DataFrame(all, index=[0])

    for df, nm in dfs[1:]:
        if len(df) == 0:
            continue
        _metrics = metrics(df)
        _metrics['name'] = nm
        res_df = pd.concat([res_df, pd.DataFrame(_metrics, index=[0])])
    
    res_df.reset_index(drop=True, inplace=True)
    return res_df

def count_contains(df, col, val, filter=False):
    print(len(df), "has", len(df[df[col].str.contains(val)]), "valid with", val, "in", col)
    if filter:
        return df[df[col].str.contains(val)]
    
def simple_fix_response(df):
    df['response'] = df.apply(lambda x: "NON INCLUSIVO" if "NON INCLUSIVO" in x['response'] else "INCLUSIVO", axis=1)
    return df