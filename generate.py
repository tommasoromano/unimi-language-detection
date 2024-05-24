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
  'Il tuo obbiettivo è identificare se il testo ed in particolare ai nomi di professioni si riferiscono a maschile o femminile o neutro.',
]
e_prompts = [
  'Ad esempio, impiegat*, impiegato/a, e impiegatÃ sono tutti esempi di parole neutre.',
]
q_prompts = [
    'Questa è la domanda: "Il testo si riferisce a maschile o femminile o neutro?"',
]
t_ttl_prompts = [
  'Puoi rispondere solamente con "maschile", "femminile" o "neutro".',
]

zsl_ttl_prompts = [
    'Sei un assistente che legge un testo e identifica se si sta riferendo ad un pubblico maschile o femminile o neutro. Puoi rispondere solamente con "maschile", "femminile" o "neutro".',
    'Sei un assistente che legge un testo e identifica se i nomi di professioni si riferiscono a maschile o femminile o neutro. Puoi rispondere solamente con un label "maschile", "femminile" o "neutro".',
    'Sei un assistente che legge un testo e risponde con un singolo label "maschile", "femminile" o "neutro" per il genere dei presenti nomi di professioni, altrimenti mette "neutro".',
]
zsl_tte_prompts = [
  'Sei un assistente che legge un testo e spiega in una singola frase se i nomi di professioni presenti si riferiscono a maschile o femminile o neutro.'
]
zsl_etl_prompts = [
    "Sei un assistente che legge una spiegazione e identifica se si riferisce a maschile o femminile o neutro. Puoi rispondere solamente con un label 'maschile', 'femminile' o 'neutro'.",
]

jobs = pd.read_csv('data/synt/jobs.csv').values.tolist()
# jobs = jobs.split('\n')
# jobs = [job.split(', ') for job in jobs]

texts = """Stiamo cercando [JOB] per la nostra azienda.
Abbiamo bisogno di [JOB] per il nostro team.
Stiamo cercndo per ricorpire il ruolo di [JOB].
Ufficio di nuova aprtura cerca [JOB] con esperienza.
Ho bisogno di [JOB] per il mio progetto.
Orario per [JOB]: monte ore variabile da 20 a 36 ore settimanali, in base alla disponibilità.
Manpower filiale di Mondovì è alla ricerca di [JOB] per EVENTI in tutta la provincia di Cuneo.
B-Free Entertainment, azienda Leader nel settore dell'intrattenimento turistico da oltre 15 anni, seleziona [JOB], con o senza esperienza per L'italia, Turchia, Tunisia ed Egitto.
Assumiamo [JOB] con il contratto regolato dal CCNL dell'animazione turistica.
Ristorazione collettiva Jobtech, agenzia per il lavoro 100 digitale, è alla ricerca di [JOB] Mensa a Milano, in zona Linate, per conto di una nota azienda operante nel settore della ristorazione collettiva, per una sostituzione nelle giornate di* giovedì 22* e* venerdì 23*."""
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

def make_synt_data_jobs(jobs, texts):
  """Generate synthetic data for job search task.
  Returns a list of [text, label] pairs."""
  data = []
  for job in jobs:
    for txt in texts:
      data.append([txt.replace('[JOB]', job[0]), job[1]])
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
    res_df = pd.DataFrame({'text':[],'ground_truth':[],'prediction':[],'prompt_id':[]})
    print("Created new dataframe.")
  count = 0
  for prompt_id, prompt in prompts:
    df_prompt = res_df[res_df['prompt_id'] == prompt_id]
    for text, genre in data:
      if text in df_prompt['text'].values:
        continue
      response = ollama.chat(model=model, messages=[
        {
          'role': 'system',
          'content': prompt,
        },
        {
          'role': 'user',
          'content': text,
        },
      ])
      row = [text, genre, response['message']['content'], prompt_id]
      res_df.loc[len(res_df)] = row
      
      count += 1
      if count % 50 == 0:
        print(f"Count: {count}", row[1:])
        res_df.to_csv(res_df_dir + '/' + res_df_name, index=False)

def make_prompts(c, i, e, q, t, formats:list[str]):
  """Generate prompts from the given components."""
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

if __name__ == '__main__':
  # Generate synthetic data and run models
  data = make_synt_data_jobs(jobs, texts)
  prompts = make_prompts(c_prompts, i_prompts, e_prompts, q_prompts, t_ttl_prompts, ['cieq','ciet','cieqt','ceiq'])
  model = models[0]
  genereate(data, model, f"fsl_ttl_{model.split(':')[0]}_synt", prompts)