import ollama
import pandas as pd
import os

# ollama run llama3:instruct
# ollama run mistral:instruct
# ollama run gemma:instruct
# ollama run phi3:instruct

c_prompts = [
  'Sei un assistente che legge ed analizza un testo italiano.',
]
i_prompts = [
  'Il tuo obbiettivo è identificare se il testo ed in particolare se i nomi di professioni si riferiscono a maschile o femminile o neutro.',
  'Il tuo obbiettivo è di identificare se il testo si riferisce a maschile o femminile o neutro (entrambi).',
]
e_prompts = [
  'Ad esempio, impiegat*, impiegato/a, e impiegatÃ sono tutti esempi di parole neutre.',
  'Per esempio, parole che contengono *, /, o Ã sono parole neutre.',
]
q_prompts = [
    'Questa è la domanda: "Il testo si riferisce a maschile o femminile o neutro?"',
]
t_prompts = [
  'Puoi rispondere solamente con un label "maschile", "femminile" o "neutro".',
  'Rispondi con un singolo label "maschile", "femminile" o "neutro" per il genere dei presenti nomi di professioni, altrimenti mettere "neutro".',
  'Classifica il testo in base al genere maschile o femminile o neutro.',
  'Rispondi tenendo la struttura "genere" + "spiegazione"'
  'Classifica il testo in base al genere maschile o femminile o neutro con struttura "genere" + "spiegazione".',
]

zsl_ttl_prompts = [
    'Sei un assistente che legge un testo e identifica se si sta riferendo ad un pubblico maschile o femminile o neutro. Puoi rispondere solamente con "maschile", "femminile" o "neutro".',
    'Sei un assistente che legge un testo e identifica se i nomi di professioni si riferiscono a maschile o femminile o neutro. Puoi rispondere solamente con un label "maschile", "femminile" o "neutro".',
    'Sei un assistente che legge un testo e risponde con un singolo label "maschile", "femminile" o "neutro" per il genere dei presenti nomi di professioni, altrimenti mettere "neutro".',
    'Classifica il testo in base al genere maschile o femminile o neutro.',
    'Classifica il testo in base al genere maschile o femminile o neutro con struttura "genere" + "spiegazione".',
]
zsl_ttl_prompts = [[f"zsl-ttl-{i}",p] for i,p in enumerate(zsl_ttl_prompts)]

zsl_tte_prompts = [
  'Sei un assistente che legge un testo e spiega in una singola frase se i nomi di professioni presenti si riferiscono a maschile o femminile o neutro.',
  'Spiega in una sola frase se il genere del testo è maschile o femminile o neutro.',
]
zsl_tte_prompts = [[f"zsl-tte-{i}",p] for i,p in enumerate(zsl_tte_prompts)]

zsl_etl_prompts = [
    "Sei un assistente che legge una frase contenente una spiegazione di un testo per identificare se si riferisce a genere maschile o femminile o neutro. Il tuo obbiettivo è identificare la risposta e capire il genere. Puoi rispondere solamente con uno dei label 'maschile', 'femminile' o 'neutro'.",
]

texts = """Stiamo cercando [JOB] per la nostra azienda.
Ufficio di nuova aprtura cerca [JOB] con esperienza.
Orario per [JOB]: monte ore variabile da 20 a 36 ore settimanali, in base alla disponibilità.
Manpower filiale di Mondovì è alla ricerca di [JOB] in tutta la provincia di Cuneo.
B-Free Entertainment, azienda Leader nel settore da oltre 15 anni, seleziona [JOB], con o senza esperienza per L'italia, Turchia, Tunisia ed Egitto.
Come [JOB] sarà in un contesto dinamico e fortemente meritocratico e dal team specializzato per facilitarne e favorirne l'inserimento.
[JOB] senza esperienza, l'agenzia offre la possibilità di imparare la professione, attraverso un percorso di formazione ed affiancamento retribuiti e a carico dell'azienda, con il quale imparare le skills e le conoscenze necessarie.
Per nuovo ufficio in Milano centro, che ha aperto le sue porte il primo Marzo 2024, siamo alla ricerca di [JOB] con ottime doti relazionali ed un network sviluppato nel cuore della città, per contribuire al nostro successo nel settore.
Ristorazione collettiva Jobtech, agenzia per il lavoro 100 digitale, è alla ricerca di [JOB] a Milano, in zona Linate, per conto di una nota azienda operante nel settore, per una sostituzione nelle giornate di* giovedì 22* e* venerdì 23*."""
texts = texts.split('\n')
# texts = [text.split(', ') for text in texts]
# texts = [[text,] for text in texts]

models = [
  'llama3:instruct',
  'mistral:instruct',
  'gemma:instruct',
  'phi3:instruct'
  ]

# !wget https://github.com/fatemehmohammadi1995/DiscriminationDetection_UNIMI/raw/main/prompts.xlsx
# !wget https://github.com/samdsk/jobsearchapi/raw/dev/db/dbdump_testdb.jobs_v5.json

def make_synt_data(replacements, replacement_tokens, label_id, target_id, texts):
  """Generate synthetic data for job search task.
  - replacements: list of [job, genre] pairs.
  - replacement_tokens: list of corresponding tokens ["[JOB]","[GENRE]"] for replacements of texts.
  - label_id: id of the label, es 1 for genre.
  - target_id: id of the target, es 0 for job.
  Returns a list of [text, label, target] pairs."""
  data = []
  for ln in replacements:
    for txt in texts:
      text = txt
      for tkn_i, tkn in enumerate(replacement_tokens):
        text = text.replace(tkn, ln[tkn_i])
      data.append([text, ln[label_id], ln[target_id]])
  return data

def genereate(data, model, res_df_name, prompts):
  """Generate predictions for the given data using the given model and prompts.
  - data: list of (text, genre) pairs.
  - prompts: are list of (id, prompt) pairs."""
  res_df_dir = 'results'
  res_df_name = res_df_name + '.csv'
  if res_df_name in os.listdir(res_df_dir):
    res_df = pd.read_csv(res_df_dir + '/' + res_df_name)
    print(f"Loaded {len(res_df)} rows.")
  else:
    res_df = pd.DataFrame({'text':[],'ground_truth':[],'target':[],'prediction':[],'prompt_id':[]})
    print("Created new dataframe.")
  count = 0
  for prompt_id, prompt in prompts:
    df_prompt = res_df[res_df['prompt_id'] == prompt_id]
    for text, genre, target in data:
      if text in df_prompt['text'].values:
        continue
      response = ollama.chat(model=model, messages=[
        { 'role': 'system', 'content': prompt, },
        { 'role': 'user', 'content': text, },
      ])
      row = [text, genre, target, response['message']['content'], prompt_id]
      res_df.loc[len(res_df)] = row
      
      count += 1
      if count % 50 == 0:
        print(f"Count: {count}", row[1:])
        res_df.to_csv(res_df_dir + '/' + res_df_name, index=False)

def make_prompts(c, i, e, q, t, formats:list[str]):
  """Generate prompts from the given components. Returns [id, prompt] pairs."""
  x = {
    'c': c,
    'i': i,
    'e': e,
    'q': q,
    't': t,
  }
  def add_prompts(letter, prompts):
    return [
      (f"{p_id}_{letter}{x_i}", p_p + ' ' + x_p)
      for p_id, p_p in prompts
      for x_i, x_p in enumerate(x[letter])
    ]
  final_prompts = []
  for format in formats:
    prompts = [(f"{format[0]}{i}", p) for i,p in enumerate(x[format[0]])]
    for letter in format[1:]:
      prompts = add_prompts(letter, prompts)
    final_prompts += prompts
  return final_prompts

def from_explanations(res_df_name):
  # two options, genereate synthetic explanation data
  # or use explanation of generated responses 
  # without knowing the real genre of the explanation
  # but the truth of the inital data
  # solution
  # take predictions, if prompt not _t then take explanation
  # we also want to understand the target is correct
  res_df_dir = 'results'
  res_df_name = res_df_name + '.csv'
  res_df = pd.read_csv(res_df_dir + '/' + res_df_name)
  for i,val in enumerate(res_df.values.tolist()):
    text, ground_truth, prediction, prompt_id = val
    if '_t' not in prompt_id:
      explanation = prediction


if __name__ == '__main__':

  if False:
    df = pd.read_json('data/dbdump_testdb.jobs_v5.json')
    titles = df.apply(lambda x: x['title'].lower(), axis=1).unique()
    # for title in titles:
    #   print(title)
    for d in df.iloc[34:45]['description']:
      print('##################')
      print(d)
    exit()

  # Generate synthetic data and run models
  jobs = pd.read_csv('data/synt/jobs.csv').values.tolist()
  data = make_synt_data(jobs, ['[JOB]'], 1, 0, texts)
  prompts = make_prompts(c_prompts, i_prompts, e_prompts, q_prompts, t_prompts, ['cieq','ciet','cieqt','ceiq'])
  prompts += zsl_ttl_prompts
  model = models[0]
  res_df_name = f"{model.split(':')[0]}_synt"
  genereate(data, model, res_df_name, prompts)
