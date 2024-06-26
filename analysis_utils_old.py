import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as skm

count_columns = lambda df, cols: df.groupby(cols).size().reset_index(name='counts').sort_values('counts', ascending=False)

filter_full_prompts = lambda df, ps: df[df['prompt_id'].isin(ps)]

def filter_zsl_prompts(df, prompt_ids_subpart):
    res_df = pd.DataFrame()
    for p in prompt_ids_subpart:
        _df = df[df['prompt_id'].str.contains(f"_{p}_") | df['prompt_id'].str.endswith(f"_{p}")]
        res_df = pd.concat([res_df, _df])
    return res_df

def filter_target(df, targets): 
    if type(targets) == str:
        targets = [targets,]
    res_df = pd.DataFrame()
    for t in targets:
        _df = df[df['target'].str.contains(t)]
        res_df = pd.concat([res_df, _df])
    return res_df

LABELS = ['maschile', 'femminile', 'neutro']

def preprocess_label_df(_df):
    df = _df.copy()
    def improve(_x):
        x = _x['prediction']
        x = str(x).replace('"','').replace('*','').lower()
        if "maschile/femminile" in x:
            return "neutro"
        if "femminile/maschile" in x:
            return "neutro" 
        if "femminile/neutro" in x:
            return "neutro"
        if "maschile/neutro" in x:
            return "neutro"
        if "entrambi" in x:
            return "neutro"
        if x not in LABELS:
            return "none"
        return x
    #f['prediction'] = df.apply(lambda x: improve(x), axis=1)
    
    # specific c0_i0_e0_t2, c0_i0_e1_t3
    def extrapolate(row):
        x = row['prediction'].lower().replace('"','')
        wrd = x
        for w in [
                'si riferisce al genere ',
                    'genere: ',
                    'classificazione: ',
                    'lassificato come ',
                    'lassificato come: ',
                    'riferisce a un genere ',
                    'riferisce a ',
                    'riferisce al ',
                    'è di genere ',
                    'con genere ',
                    'riferito al genere ',
                    'la risposta è: ',
                ]:
            if w in x:
                wrd = x.split(w)[1]#.split(' ')[0]
                break
            
        for w in [
            'genere ',
            'risposta: ',
        ]:
            if wrd.startswith(w):
                wrd = wrd.split(' ')[1]

        for w in [
            'maschile/femminile',
            'maschile e femminile',
            'femminile e maschile',
            'femminile e neutro',
            'maschile o femminile',
            'femminile o neutro',
            'maschile o neutro',
            'femminile/maschile',
            'femminile/neutro',
            'neutro/femminile',
            'entramb',
            'neutro/entrambi',
            'm/f',
            'misto',
            'il testo si riferisce sia ',
        ]:
            if wrd.startswith(w):
                return 'neutro'
            
        for l in LABELS:
            if wrd.startswith('**'+l):
                return l
            if wrd.startswith('il testo è '+l):
                return l
            if wrd.startswith(l):
                for e in ['\n','*','.',';',',',':',
                        ' poich',
                        f" ({row['target']})",
                        '/spiegazione',
                        ' in italiano',
                        ' (entrambi).',
                        ' (entrambi),',
                        ]:
                    if wrd.startswith(l+e):
                        return l
                try:
                    idx = wrd.index(')')
                    if len(wrd) == idx+1:
                        return l
                    if wrd[idx+1] == '\n':
                        return l
                    if wrd[idx+1] == '.':
                        return l
                except:
                    pass
        return wrd
    # df['prediction'] = df.apply(lambda x: extrapolate(x), axis=1)

    def choose(x):
        p = x['prompt_id']
        if '_t0' in p or '_t1' in p:
            return improve(x)
        return extrapolate(x)

    df['prediction'] = df.apply(lambda x: choose(x), axis=1)

    df = df[df['prediction'].isin(LABELS)]
    return df

def valuate_label_results(df):
    accuracy = skm.accuracy_score(df['ground_truth'], df['prediction'])
    f1 = skm.f1_score(df['ground_truth'], df['prediction'], average='weighted')
    precision = skm.precision_score(df['ground_truth'], df['prediction'], average='weighted')
    recall = skm.recall_score(df['ground_truth'], df['prediction'], average='weighted')
    # res = multilabel_confusion_matrix(df['ground_truth'], df['prediction'], labels=LABELS).ravel()
    # print(res)
    report = skm.classification_report(df['ground_truth'], df['prediction'], target_names=LABELS)
    # print(report)
    # tn, fp, fn, tp = res
    # accuracy = (tp + tn) / (tp + tn + fp + fn)
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # f1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, f1, precision, recall

def valuation_table(df, target_filters, inclusive=False):
    df_count = len(df)
    df = preprocess_label_df(df)
    df_count_after = len(df)

    if inclusive:
        df['ground_truth'] = df['ground_truth'].apply(lambda x: "inclusive" if x == "neutro" else "non inclusive")
        df['prediction'] = df['prediction'].apply(lambda x: "inclusive" if x == "neutro" else "non inclusive")

    res = pd.DataFrame({'target_filter':[], 'samples':[],'accuracy': [], 'f1': [], 'precision': [], 'recall': []})
    res.loc[len(res)] = [None,df_count_after/df_count] + list(valuate_label_results(df))
    for t in target_filters:
        _df = filter_target(df, t)
        res.loc[len(res)] = [t,len(_df)/df_count] + list(valuate_label_results(_df))
    return res

def analyze_label_results(df):

    sns.countplot(data=df, x='ground_truth')
    plt.show()

    print('Dataframe count:', len(df))
    df = preprocess_label_df(df)
    print('Dataframe count after preprocess:', len(df))

    sns.countplot(data=df, x='prediction')
    plt.show()

    df_pivot = df.pivot_table(index='ground_truth', columns='prediction', values='prompt_id', aggfunc='count', fill_value=0)
    sns.heatmap(df_pivot, annot=True, fmt='d')
    plt.show()

    cm = skm.confusion_matrix(df['ground_truth'], df['prediction'], normalize='true')
    disp = skm.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot()
    plt.show()

    accuracy, f1, precision, recall = valuate_label_results(df)

    plt.table(cellText=[[accuracy, f1, precision, recall]], colLabels=['Accuracy', 'F1', 'Precision', 'Recall'], cellLoc='center', loc='bottom')
    plt.axis('off')
    plt.show()
    
    return count_columns(df, ['prediction'])

neutrals = ['giornaista',
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
            'igienista dentale']