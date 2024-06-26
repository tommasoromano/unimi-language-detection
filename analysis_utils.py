import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as skm

def group_count(df, by, n=10, hue=None, others=True):

    def gc(df, h=None):
        # def fix_col(c):
        #     if 'proportion' in c.columns:
        #         c = c.rename({'count':'proportion'}, axis=1)
        #     return c
        _c = df[[by]] if h is None else df[[by,hue]]
        if n <= 0:
            return _c.value_counts().sort_values(ascending=False).to_frame().reset_index()
        else:
            c = _c.value_counts().sort_values(ascending=False)[:n].to_frame().reset_index()
        if df.shape[0] - c['count'].sum() > 0 and others:
            if h is None:
                c.loc[len(c)] = ['OTHERS', df.shape[0] - c['count'].sum()]
            else:
                c.loc[len(c)] = ['OTHERS', h, df.shape[0] - c['count'].sum()]
        
        return c
    
    if hue is not None:
        c = None
        for h in df[hue].unique():
            if c is None:
                c = gc(df[df[hue] == h], h=h)
                c['proportion_hue'] = c['count'] / df[df[hue] == h].shape[0]
            else:
                _c = gc(df[df[hue] == h], h=h)
                _c['proportion_hue'] = _c['count'] / df[df[hue] == h].shape[0]
                c = pd.concat([c, _c])
        c = c.reset_index(drop=True)
        c['proportion_by'] = c.apply(lambda x: x['count'] / df[df[by] == x[by]].shape[0], axis=1)
    else:
        c = gc(df)
        c['proportion'] = c['count'] / df.shape[0]

    return c

def plot_df(df, by, n=10, hue=None, others=True, title='', plots='012', count='count'):

    df = df.copy()
    if title != '':
        title = f" - {title}"
    # if hue is not None and hue_contains is not None:
    #     df = df[df[hue].str.contains(hue_contains)]

    n_str = '' if n <= 0 else f"{n}"

    if '0' in plots:
        c = group_count(df, by, n=n, others=others)
        c = c.sort_values(by=count, ascending=False)
        sns.barplot(data=c, x=count, y=by)
        plt.title(f"Top {n_str} {by}{title}")
        plt.show()

    if '1' in plots:
        cs = group_count(df, by, n=-1, others=others)[count].cumsum()
        ax = sns.lineplot(cs)
        ax.set_xticks([])
        plt.title(f'{by} cumulatively{title}')
        plt.show()

    if hue is None:
        return
    if '2' in plots:
        c = group_count(df, by, n=n, hue=hue, others=others)
        c = c.sort_values(by=count, ascending=False)
        if hue == 'response':
            palette ={"neutral": "grey", "male": "C0", "female": "C3"}
            sns.barplot(data=c, x=count, y=by, hue=hue, palette=palette)
        else:
            sns.barplot(data=c, x=count, y=by, hue=hue)
        plt.title(f"Top {n_str} {by} by {hue}{title}")
        plt.show()

def not_valid(df, valids:list[str]):
    return df[~df['response'].isin(valids)]

def fix_responses(_df):
    df = _df.copy()
    def f(r):
        W = ['i','you','someone','the','neither','one','he/she','he/she/they','he/she/it']
        r = r.replace('"','').replace(':','').replace('*','').replace('[','').replace(']','')
        tkn = '**'
        for ln in [
            f"refers to {tkn}",
            f"refers to a {tkn}",
            f"referred to is {tkn}",
            f"referred to {tkn}",
            f"would be {tkn}",
            f"answer is {tkn}",
            f"is\n\n{tkn}",
            f"fill is {tkn}",
            f"person as {tkn}",
            f"person is {tkn}",
            f"person with {tkn}",
            f"{tkn}'s",
            f"{tkn}'re",
            f"{tkn} has a job",
            f"{tkn} was a",
            f"{tkn} were a",
            f"{tkn} will",
            f"{tkn} has always",
            f"{tkn} has studied",
            f"{tkn} is studying",
            ]:
            for v in VALID:
                if ln.replace(tkn, v) in r:
                    return v
            for w in W:
                if ln.replace(tkn, w) in r:
                    return 'neutral'
        if r in W:
            return 'neutral'
        if 'both' in r:
            return 'both'
        return r
    df['response'] = df.apply(lambda x: f(x['response']), axis=1)
    return df

def normalize_labels(df):
    df = df.copy()
    def f(r):
        if r in VALID:
            if r == 'he' or r == 'male':
                return 'male'
            if r == 'she' or r == 'female':
                return 'female'
            # return r
            return 'neutral'
        else:
            return None
    df['response'] = df.apply(lambda x: f(x['response']), axis=1)
    df = df.dropna()
    return df

def plot_compare_df(original, fixed, by):
    df_fix_cmp = pd.DataFrame({by:[],"df":[],"count":[]})
    for p in original[by].unique():
        df_fix_cmp = pd.concat([df_fix_cmp, pd.DataFrame({by:[p],"df":["original"],"count":[original[original[by] == p].shape[0]]})])
        df_fix_cmp = pd.concat([df_fix_cmp, pd.DataFrame({by:[p],"df":["fixed"],"count":[fixed[fixed[by] == p].shape[0]]})])
    ax = sns.barplot(data=df_fix_cmp, y='count', x=by, hue='df')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title(f'Original vs Fixed - {by}')
    plt.show()

def df_filter(df, by, contains):
    return df[df[by].str.contains(contains)]

def pivot_df(df, index, on, value):
    res = None
    for j in df[index].unique():
        _df = {index: [j],}
        _df.update({r: [None] for r in df[on].unique()})
        for i,r in df[df[index] == j].iterrows():
            _df[r[on]] = r[value]
        if res is None:
            res = pd.DataFrame(_df)
        else:
            res = pd.concat([res, pd.DataFrame(_df)])
    return res

def valuate_label_results(df, ground_truth, prediction):
    LABELS = df[ground_truth].unique()
    accuracy = skm.accuracy_score(df[ground_truth], df[prediction])
    f1 = skm.f1_score(df[ground_truth], df[prediction], average='weighted')
    precision = skm.precision_score(df[ground_truth], df[prediction], average='weighted')
    recall = skm.recall_score(df[ground_truth], df[prediction], average='weighted')
    # res = multilabel_confusion_matrix(df[ground_truth], df[prediction], labels=LABELS).ravel()
    # print(res)
    report = skm.classification_report(df[ground_truth], df[prediction], target_names=LABELS)
    # print(report)
    # tn, fp, fn, tp = res
    # accuracy = (tp + tn) / (tp + tn + fp + fn)
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # f1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, f1, precision, recall

def valuation_table(df, by, ground_truth, prediction):
    res = pd.DataFrame({by:[],'accuracy': [], 'f1': [], 'precision': [], 'recall': []})
    for _by in df[by].unique():
        _df = df[df[by] == _by]
        accuracy, f1, precision, recall = valuate_label_results(_df, ground_truth, prediction)
        res = pd.concat([res,
                         pd.DataFrame({
                                by:[_by],
                                'accuracy': [accuracy],
                                'f1': [f1],
                                'precision': [precision],
                                'recall': [recall]
                            })])
    res = res.sort_values(by='accuracy', ascending=False)
    return res

def analyze_label_results(df, ground_truth, prediction, title=''):

    LABELS = df[ground_truth].unique()
    if title != '':
        title = f" - {title}"

    # sns.countplot(data=df, x=ground_truth)
    # plt.show()

    # print('Dataframe count:', len(df))
    # df = preprocess_label_df(df)
    # print('Dataframe count after preprocess:', len(df))

    # sns.countplot(data=df, x=prediction)
    # plt.show()

    df_pivot = df.pivot_table(index=ground_truth, columns=prediction, values='prompt_id', aggfunc='count', fill_value=0)
    sns.heatmap(df_pivot, annot=True, fmt='d')
    plt.title(f'Confusion Matrix{title}')
    plt.show()

    cm = skm.confusion_matrix(df[ground_truth], df[prediction], normalize='true')
    disp = skm.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot()
    plt.title(f'Confusion Matrix{title}')
    plt.show()

    accuracy, f1, precision, recall = valuate_label_results(df, ground_truth, prediction)

    plt.table(cellText=[[accuracy, f1, precision, recall]], colLabels=['Accuracy', 'F1', 'Precision', 'Recall'], cellLoc='center', loc='bottom')
    plt.axis('off')
    plt.show()
    
    # return count_columns(df, [prediction])