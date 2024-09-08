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
        if r.startswith('NON INCLU'):
            return 'NON INCLUSIVO'
        elif r.startswith('INCLU'):
            return 'INCLUSIVO'
        if r.startswith(' NON INCLU'):
            return 'NON INCLUSIVO'
        elif r.startswith(' INCLU'):
            return 'INCLUSIVO'
        if '"NON INCLUSIVO"' in r:
            return 'NON INCLUSIVO'
        elif '"INCLUSIVO"' in r:
            return 'INCLUSIVO'
        if 'NON È INCLU' in r:
            return 'NON INCLUSIVO'
        elif 'È NON INCLU' in r:
            return 'NON INCLUSIVO'
        if ': INCLU' in r:
            return 'INCLUSIVO'
        elif ': NON INCLU' in r:
            return 'NON INCLUSIVO'
        if '\n\nINCLU' in r:
            return 'INCLUSIVO'
        elif '\n\nNON INCLU' in r:
            return 'NON INCLUSIVO'
        if '**INCLU' in r:
            return 'INCLUSIVO'
        elif '**NON INCLU' in r:
            return 'NON INCLUSIVO'
        if '** INCLU' in r:
            return 'INCLUSIVO'
        elif '** NON INCLU' in r:
            return 'NON INCLUSIVO'
        elif 'È INCLU' in r:
            return 'INCLUSIVO'
        if "ENTRAMBI" in r:
            return 'INCLUSIVO'
        return None

    df_fix = df.copy()
    
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

def make_df_len(df:pd.DataFrame, groups=10):
    # df['len'] = df.apply(lambda x: len(x['text']), axis=1)
    mn = 0 # df['len'].min()
    mx = 300 # df['len'].max()
    step = (mx - mn) // groups
    df['len_group'] = df.apply(lambda x: f"{((x['len']+step)//step)*step}", axis=1)
    return df

def make_df_multi_metrics(df_metrics:pd.DataFrame, metrics=[]):
    df_metrics = df_metrics.copy()
    if len(metrics) != 0:
        df_metrics = df_metrics[["name"] + metrics]
    res_df_melt = df_metrics.melt(id_vars=['name'], var_name='metric', value_name='value')
    return res_df_melt

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

def make_all_dfs(
        models:list[str]=[
            "phi3",
    "phi3-finetuned",
    "llama3",
    "mistral",
    "gemma2",
    "qwen2",
    ],
    is_seed:bool=False,
):
    dfs_all = []
    dfs_by_len = []
    dfs_by_prompt = []
    dfs_by_pl = []
    # df = pd.read_csv(f'results/{model}_split-long-v0.csv')
    for model in models:
        print(model)
        try:
            df = pd.read_csv(sc_csv_name(model, False, is_seed))
            df = fix_df(df)
            # df = count_contains(df, "response", "INCLUSIVO", True)
            # df = simple_fix_response(df)
            df = fix_df_model_response(df, model)
            # confusion_matrix(df)
            df = make_df_len(df)
            dfs_all.append((df,model))
            for p in df['prompt_id'].unique():
                df_ = df[df['prompt_id'] == p]
                dfs_by_prompt.append((df_,f"{model}_{p}"))
            for l in df['len_group'].unique():
                df_ = df[df['len_group'] == l]
                dfs_by_len.append((df_,f"{model}_{l}"))
            for p in df['prompt_id'].unique():
                df_ = df[df['prompt_id'] == p]
                for l in df['len_group'].unique():
                    df__ = df_[df_['len_group'] == l]
                    dfs_by_pl.append((df__,f"{model}_{p}_{l}"))
        except Exception as e:
            print(model, e)
    return dfs_all ,dfs_by_len ,dfs_by_prompt ,dfs_by_pl 

def get_best_prompt(
        dfs:list[tuple[pd.DataFrame,str]],
        metric='precision',
        ):
    df_metrics = metrics_of_dfs(dfs).sort_values(metric, ascending=False)
    res = dict()
    for row in df_metrics.iterrows():
        nm = row[1]['name']
        model = nm.split('_')[0]
        prompt = nm.split('_')[1]
        performance = row[1][metric]
        if model not in res:
            res[model] = (prompt, 0)
        if performance > res[model][1]:
            res[model] = (prompt, performance)
    return res

def confusion_matrix(df, true_col='true', pred_col='response'):
    cm = skm.confusion_matrix(df[true_col], df[pred_col], labels=LABELS, normalize='true')
    disp = skm.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot()
    plt.show()

def merge_dfs(dfs:list[tuple[pd.DataFrame,str]]):
    res_df = dfs[0][0].copy()
    res_df["model"] = dfs[0][1]

    for df, nm in dfs[1:]:
        if len(df) == 0:
            continue
        _df = df.copy()
        _df["model"] = nm
        res_df = pd.concat([res_df, _df])
    
    res_df.reset_index(drop=True, inplace=True)
    return res_df

def plot_compare_matrix(dfs:list[tuple[pd.DataFrame,str]]):
    df_metrics = metrics_of_dfs(dfs)
    df = make_df_multi_metrics(df_metrics, [
        'true_positives',
        'true_negatives',
        'false_positives',
        'false_negatives',
    ])
    sns.barplot(data=df, x='metric', y='value', hue='name')
    plt.title(f"Confusion Matrix Comparison")
    plt.show()

def heatmap_of_performance(
        dfs:list[tuple[pd.DataFrame,str]],
        title="Models"
        ):
    df = metrics_of_dfs(dfs)[[
        'name',
        'true_positives',
        'true_negatives',
        'false_positives',
        'false_negatives',
        'recall',
        'specificity',
        'accuracy',
        'precision',
        'f1',
    ]]
    df.set_index('name', inplace=True)
    # df = df.T

    # Create a heatmap
    plt.figure(figsize=(8, max(6,0.2*len(dfs))))
    sns.heatmap(df, annot=True, cmap='coolwarm', linewidths=0.5)

    # Add title and labels
    plt.title('Precision, Recall, and F1 Score Heatmap')
    plt.xlabel('Metric')
    plt.ylabel('Model')

    # Ensure the plot doesn't get cut off
    plt.tight_layout()
    plt.title("Heatmap Performance of "+title)
    plt.show()

def heatmap_of_prompts(
        dfs:list[tuple[pd.DataFrame,str]],
        metric_col="precision",
        title="Models"
        ):
    df_metrics = metrics_of_dfs(dfs)
    df_metrics = df_metrics[~df_metrics['name'].str.contains('finetuned')]
    df_metrics['prompt'] = df_metrics.apply(lambda x: x['name'].split('_')[1], axis=1)
    df_metrics['name'] = df_metrics.apply(lambda x: '_'.join([s for i,s in enumerate(x['name'].split('_')) if i != 1]), axis=1)
    df = df_metrics.pivot(index='name', columns='prompt', values=metric_col)
    plt.figure(figsize=(8, max(6,0.2*len(dfs))))
    sns.heatmap(df, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Heatmap Performance by Prompt of "+title)
    plt.show()

def plot_prompts(
        dfs:list[tuple[pd.DataFrame,str]], 
        metric_col="precision",
        title="Models"
        ):
    df_metrics = metrics_of_dfs(dfs)
    df_metrics = df_metrics[~df_metrics['name'].str.contains('finetuned')]
    df_metrics['prompt'] = df_metrics.apply(lambda x: x['name'].split('_')[1], axis=1)
    df_metrics['name'] = df_metrics.apply(lambda x: '_'.join([s for i,s in enumerate(x['name'].split('_')) if i != 1]), axis=1)
    sns.barplot(
        data=df_metrics[["name","prompt",metric_col]],
        x="prompt", y=metric_col, hue="name",
    )
    plt.title(f"Performance by Prompt of {title}")
    plt.show()

def plot_len_metrics(
        dfs:list[tuple[pd.DataFrame,str]], 
        metric_col,
        split_pos,
        title="Models"
        ):
    df_metrics = metrics_of_dfs(dfs)
    df_metrics['len_group'] = df_metrics.apply(lambda x: int(x['name'].split('_')[split_pos]), axis=1)
    df_metrics['name'] = df_metrics.apply(lambda x: '_'.join([s for i,s in enumerate(x['name'].split('_')) if i != split_pos]), axis=1)
    sns.relplot(
        data=df_metrics[["name","len_group",metric_col]], kind="line",
        x="len_group", y=metric_col, hue="name", markers=True, dashes=False,
    )
    plt.title(f"Performance by Length of {title}")
    plt.show()

def plot_labels_by_model(df:pd.DataFrame,title=""):
    sns.countplot(data=df, x="true", hue="model")
    plt.title("Labels Count" + title)
    plt.show()

def plot_len_groups(df, title=""):
    df = df.copy()
    df['len_group'] = df.apply(lambda x: int(x['len_group']), axis=1)
    df = df.sort_values(by='len_group', ascending=True)
    sns.countplot(data=df, x='len_group', hue="model")
    plt.title("Distributions of Lengths" + title)
    plt.show()

def plot_multi_metrics(df_metrics, metrics=[]):
    df_metrics = make_df_multi_metrics(df_metrics, metrics)
    sns.barplot(data=df_metrics, x='value', y='name', hue='metric')
    plt.show()

def count_contains(df, col, val, filter=False):
    print(len(df), "has", len(df[df[col].str.contains(val)]), "valid with", val, "in", col)
    if filter:
        return df[df[col].str.contains(val)]
    
def simple_fix_response(df):
    df['response'] = df.apply(lambda x: "NON INCLUSIVO" if "NON INCLUSIVO" in x['response'] else "INCLUSIVO", axis=1)
    return df