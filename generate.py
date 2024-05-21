import ollama
import pandas as pd
import os

# ollama run llama3:instruct
# ollama run mistral:instruct
# ollama run gemma:instruct
# ollama run phi3:instruct

text_to_label_prompts = [
    'Sei un assistente che legge un testo e identifica se si sta riferendo ad un pubblico maschile o femminile o neutro. Puoi rispondere solamente con "maschile", "femminile" o "neutro".',
    'Sei un assistente che legge un testo e identifica se i nomi di professioni si riferiscono a maschile o femminile o neutro. Puoi rispondere solamente con un label "maschile", "femminile" o "neutro".',
    'Sei un assistente che legge un testo e risponde con un singolo label "maschile", "femminile" o "neutro" per il genere dei presenti nomi di professioni, altrimenti mette "neutro".',
]
text_to_explain_prompts = [
  'Sei un assistente che legge un testo e spiega in una singola frase se i nomi di professioni presenti si riferiscono a maschile o femminile o neutro.'
]
explain_to_label_prompts = [
    "Sei un assistente che legge una spiegazione e identifica se si riferisce a maschile o femminile o neutro. Puoi rispondere solamente con un label 'maschile', 'femminile' o 'neutro'.",
]

articoli = """il [JOB], maschile
lo [JOB], maschile
l' [JOB], maschile
la [JOB], femminile
l* [JOB], neutro
i [JOB], maschile
gli [JOB], maschile
le [JOB], femminile
un [JOB], maschile
uno [JOB], maschile
una [JOB], femminile
un' [JOB], femminile
un* [JOB], neutro"""
articoli = articoli.split('\n')
articoli = [articolo.split(', ') for articolo in articoli]

jobs = """programmatore, maschile
programmatrice, femminile
programmatori, maschile
programmatrici, femminile
programmatori e programmatrici, neutro
programmatori o programmatrici, neutro
programmator*, neutro
programmatorÃ, neutro
impiegato, maschile
impiegata, femminile
impiegati, maschile
impiegate, femminile
impiegati e impiegate, neutro
impiegati o impiegate, neutro
impiegat*, neutro
impiegato/a, neutro
impiegatÃ, neutro
cameriere, neutro
cameriera, femminile
camerier*, neutro
cameriere/a, neutro
camerierÃ, neutro
infermiere, neutro
infermiera, femminile
infermier*, neutro
infermiere/a, neutro
infermierÃ, neutro
farmacista, neutro
contabile, neutro
segretario, maschile
segretaria, femminile
segretar*, neutro
segretario/a, neutro
segretarÃ, neutro
direttore, maschile
direttrice, femminile
ballerina, femminile
ballerino, maschile
ballerin*, neutro
ballerino/a, neutro
ballerinÃ, neutro
giornaista, neutro
autista, neutro
dentista, neutro
ginnasta, neutro
attore, maschile
attrice, femminile
attor*, neutro
attorÃ, neutro
attore/attrice, neutro
attore e attrice, neutro
attore o attrice, neutro
dottore, maschile
dottoressa, femminile
dottor*, neutro
dottorÃ, neutro
dottore/dottoressa, neutro
dottore e dottoressa, neutro
dottore o dottoressa, neutro
architetto, maschile
architetta, femminile
architett*, neutro
architettÃ, neutro
architetto/architetta, neutro
architetto e architetta, neutro
architetto o architetta, neutro
architetto/a, neutro
candidato, maschile
candidata, femminile
candidat*, neutro
candidato/a, neutro
candidatÃ, neutro"""
jobs = jobs.split('\n')
jobs = [job.split(', ') for job in jobs]

texts = """Stiamo cercando [JOB] per la nostra azienda.
Abbiamo bisogno di [JOB] per il nostro team.
Stiamo cercndo per ricorpire il ruolo di [JOB].
Ufficio di nuova aprtura cerca [JOB] con esperienza.
Ho bisogno di [JOB] per il mio progetto.
Orario per [JOB]: monte ore variabile da 20 a 36 ore settimanali, in base alla disponibilità.
Manpower filiale di Mondovì è alla ricerca di [JOB] per EVENTI in tutta la provincia di Cuneo.
B-Free Entertainment, azienda Leader nel settore dell'intrattenimento turistico da oltre 15 anni, seleziona [JOB], con o senza esperienza per L'italia, Turchia, Tunisia ed Egitto.
Assumiamo [JOB] con il contratto regolato dal CCNL dell'animazione turistica."""
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

def make_data_jobs(jobs, texts):
  data = []
  for job in jobs:
    for txt in texts:
      data.append([txt.replace('[JOB]', job[0]), job[1]])
  return data

def genereate(data, model, res_df_name, prompts):
  if res_df_name in os.listdir():
    res_df = pd.read_csv(res_df_name)
  else:
    res_df = pd.DataFrame({'text':[],'ground_truth':[],'prediction':[],'prompt_id':[]})
  count = 0
  for prompt_id, prompt in enumerate(prompts):
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
    res_df.to_csv(res_df_name, index=False)

data = make_data_jobs(jobs, texts)
model = models[2]
genereate(data, model, f"ttl_{model.split(':')[0]}_synt.csv", text_to_label_prompts)
model = models[3]
genereate(data, model, f"ttl_{model.split(':')[0]}_synt.csv", text_to_label_prompts)