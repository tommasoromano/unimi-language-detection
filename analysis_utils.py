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

    # plot a barplot of top 10 responses by count and a column of the remaining responses
    # _df = df[df['prompt_id'] != 'zslcot#0']
    _df = df.copy()
    top = 10
    res = _df['response'].value_counts()
    res = res.reset_index()
    res.columns = ['response','count']
    res = res.sort_values(by='count', ascending=False)
    res['%'] = res['count'] / len(_df)
    res = res.reset_index(drop=True)
    res = res.head(top)
    res = pd.concat([res,pd.DataFrame({'response':['OTHER'],'count':[len(_df)-res['count'].sum()],'%':[(len(_df)-res['count'].sum())/len(_df)]})])
    # print(res)
    sns.barplot(data=res, y='response', x='%')
    plt.title(f"Top {top} responses - {model}")
    plt.show()

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
        negative_rate = true_negatives / (true_negatives + false_negatives)
    except:
        negative_rate = None
    try:
        false_positive_rate = false_positives / (false_positives + true_negatives)
    except:
        false_positive_rate = None
        
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
        'bACC': (recall + specificity) / 2 if recall is not None and specificity is not None else (recall or specificity),
        'precision': precision,
        'f1': f1,
        'negative_rate': negative_rate,
        'false_positive_rate': false_positive_rate,
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
    "phi3-finetuned",
    "gpt-4o-mini",
            "phi3",
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

DEFAULT_METRIC = "bACC"

def get_best_prompt(
        dfs:list[tuple[pd.DataFrame,str]],
        metric=DEFAULT_METRIC,
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
        'bACC',
        'precision',
        'f1',
    ]]
    df.set_index('name', inplace=True)
    # df = df.T

    # Create a heatmap
    plt.figure(figsize=(8, max(6,0.2*len(dfs))))
    sns.heatmap(df, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".3f")

    # Add title and labels
    plt.title('Precision, Recall, and F1 Score Heatmap')
    plt.xlabel('Metric')
    plt.ylabel('Model')

    # Ensure the plot doesn't get cut off
    plt.tight_layout()
    plt.title("Heatmap Performance"+title)
    plt.show()

def heatmap_of_prompts(
        dfs:list[tuple[pd.DataFrame,str]],
        metric_col=DEFAULT_METRIC,
        title="Models"
        ):
    df_metrics = metrics_of_dfs(dfs)
    # df_metrics = df_metrics[~df_metrics['name'].str.contains('finetuned')]
    df_metrics['prompt'] = df_metrics.apply(lambda x: x['name'].split('_')[1], axis=1)
    df_metrics['name'] = df_metrics.apply(lambda x: '_'.join([s for i,s in enumerate(x['name'].split('_')) if i != 1]), axis=1)
    df = df_metrics.pivot(index='name', columns='prompt', values=metric_col)
    plt.figure(figsize=(8, max(6,0.2*len(dfs))))
    sns.heatmap(df, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".3f")
    plt.title("Heatmap Performance by Prompt"+title)
    plt.show()

def plot_len_metrics(
        dfs:list[tuple[pd.DataFrame,str]], 
        split_pos,
        metric_col=DEFAULT_METRIC,
        title="Models"
        ):
    df_metrics = metrics_of_dfs(dfs)
    df_metrics['len_group'] = df_metrics.apply(lambda x: int(x['name'].split('_')[split_pos]), axis=1)
    df_metrics['name'] = df_metrics.apply(lambda x: '_'.join([s for i,s in enumerate(x['name'].split('_')) if i != split_pos]), axis=1)
    sns.relplot(
        data=df_metrics[["name","len_group",metric_col]], kind="line",
        x="len_group", y=metric_col, hue="name", style="name", markers=True, dashes=False,
        palette="tab10",
    )
    plt.title(f"Performance by Length of {title}")
    plt.show()

def plot_labels_by_model(df:pd.DataFrame,title=""):
    sns.countplot(data=df, x="true", hue="model", palette="tab10",)
    plt.title("Labels Count" + title)
    plt.show()

def plot_len_groups(df, title=""):
    df = df.copy()
    df['len_group'] = df.apply(lambda x: int(x['len_group']), axis=1)
    df = df.sort_values(by='len_group', ascending=True)
    sns.countplot(data=df, x='len_group', hue="model", palette="tab10",)
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

def heatmap_of_chars(
        dfs:list[tuple[pd.DataFrame,str]],
        metric_col=DEFAULT_METRIC,
        title="Models"
        ):
    # df_metrics = df_metrics[~df_metrics['name'].str.contains('finetuned')]
    def fn(x):
        v = x['text_JOB_value']
        l = x['text_JOB_label']
        if l == 'neutro':
            return 'neutro'
        elif "*" in v:
            return 'star'
        elif "Ã" in v:
            return 'Ã'
        elif "/" in v:
            return '/'
        elif " o " in v:
            return ' o '
        elif " e " in v:
            return ' e '
        else:
            return l
    _dfs = []
    for df, nm in dfs:
        # _df = df.copy()
        # _df['name'] = _df.apply(lambda x: fn(x), axis=1)
        _dfs.append((df[df['text_JOB_label'] == 'neutro'], nm+"_job-neutro"))
        _dfs.append((df[df['text_JOB_label'] == 'maschile'], nm+"_job-maschile"))
        _dfs.append((df[df['text_JOB_label'] == 'femminile'], nm+"_job-femminile"))
        _dfs.append((df[df['text_ADJ_label'] == 'neutro'], nm+"_adj-neutro"))
        _dfs.append((df[df['text_ADJ_label'] == 'maschile'], nm+"_adj-maschile"))
        _dfs.append((df[df['text_ADJ_label'] == 'femminile'], nm+"_adj-femminile"))
        _dfs.append((df[(df['text_JOB_value'] == '') & (df['text_ADJ_value'] == '')], nm+"_others"))
        # _dfs.append((df[df['text_ADJ_label'].str.len() > 0], nm+"_adjs"))
        df_job = df[df['text_JOB_value'].str.len() > 0]
        _dfs.append((df_job[df_job['text_JOB_value'].str.contains("\*")], nm+"_star"))
        _dfs.append((df_job[df_job['text_JOB_value'].str.contains("Ã")], nm+"_Ã"))
        _dfs.append((df_job[df_job['text_JOB_value'].str.contains("/")], nm+"_slash"))
        _dfs.append((df_job[(df_job['text_JOB_value'].str.contains(" o ")) | (df_job['text_JOB_value'].str.contains(" e "))], nm+"_cong"))
        # _dfs.append((df_job[df_job['text_JOB_value'].str.contains(" e ")], nm+"_e"))
    df_metrics = metrics_of_dfs(_dfs)
    df_metrics['chars'] = df_metrics.apply(lambda x: x['name'].split('_')[2], axis=1)
    df_metrics['name'] = df_metrics.apply(lambda x: '_'.join([s for i,s in enumerate(x['name'].split('_')) if i != 2]), axis=1)
    df = df_metrics.pivot(columns='name', index='chars', values=metric_col)
    plt.figure(figsize=(8, max(6,0.2*len(dfs))))
    sns.heatmap(df, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".3f")
    # plt.xticks(rotation=45, ha='right')
    plt.title("Heatmap Performance by Words"+title)
    plt.show()

# https://artificialanalysis.ai/

tokens_per_second = {
    "phi3-finetuned": 72,
    "gpt-4o-mini": 144,
    "gemma2": 126,
    "llama3": 106,
    "mistral": 104,
    "phi3": 72,
    "qwen2": 56,
}

input_prices_1M_tokens = {
    "phi3-finetuned": 0.14,
    "gpt-4o-mini": 0.15,
    "gemma2": 0.2,
    "llama3": 0.15,
    "mistral": 0.15,
    "phi3": 0.14,
    "qwen2": 0.35,
}

output_prices_1M_tokens = {
    "phi3-finetuned": 0.14,
    "gpt-4o-mini": 0.6,
    "gemma2": 0.2,
    "llama3": 0.2,
    "mistral": 0.2,
    "phi3": 0.14,
    "qwen2": 0.4,
}

models_quality = {
    "phi3-finetuned": 0.991,
    "gpt-4o-mini": 0.888,
    "gemma2": 0.598,
    "llama3": 0.563,
    "mistral": 0.520,
    "phi3": 0.508,
    "qwen2": 0.580,
}

# 3:1 ratio
prices_1M_tokens = {
    k: 3*input_prices_1M_tokens[k] + output_prices_1M_tokens[k]
    for k in input_prices_1M_tokens
}

def plot_general_costs(
        
):
    df = pd.DataFrame({
        'model': list(prices_1M_tokens.keys()),
        'price': list(prices_1M_tokens.values()),
        'speed': list(tokens_per_second.values()),
        'quality': list(models_quality.values()),
    })
    df = df.sort_values(by='price', ascending=True)
    sns.barplot(data=df, x='model', y='price')
    plt.title("Prices per 1M tokens")
    plt.xticks(rotation=45, ha='right')
    plt.show()
    df = df.sort_values(by='speed', ascending=False)
    sns.barplot(data=df, x='model', y='speed')
    plt.title("Output tokens per second")
    plt.xticks(rotation=45, ha='right')
    plt.show()
    ax = sns.scatterplot(data=df, x='speed', y='price', hue='model', style="model", s=100)
    plt.title("Price vs Speed")
    ax.fill_between([100, 150], 0.5, 1.0, color='green', alpha=0.1)
    ax.fill_between([50, 100], 1.0, 1.5, color='red', alpha=0.1)
    plt.legend(loc = "upper center")
    plt.show()
    ax = sns.scatterplot(data=df, x='quality', y='price', hue='model', style="model", s=100)
    plt.title("Price vs Quality")
    ax.fill_between([0.75, 1.0], 0.5, 1.0, color='green', alpha=0.1)
    ax.fill_between([0.5, 0.75], 1.0, 1.5, color='red', alpha=0.1)
    plt.legend(loc = "upper right")
    plt.show()