import ollama
import pandas as pd
import os
from nlp_synt_data import *
from data.prompts import PROMPTS_JOB_CIEQT_V0

# ollama run llama3:instruct
# ollama run mistral:instruct
# ollama run gemma:instruct
# ollama run phi3:instruct

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
  prompts = PromptGenerator.generate(PROMPTS_JOB_CIEQT_V0,[
                                      ['c','i','e','q'],
                                      ['c','i','e','t'],
                                      ['c','i','e','q','t'],
                                      ['c','e','i','q'],
                                      ])
  
  # jobs = pd.read_csv('data/synt/jobs.csv').values.tolist()
  # data = DataGenerator.generate([(t,'none') for t in texts], {'JOB': [j for j in jobs]})

  texts = pd.read_csv('data/job_description_seed_dataset_improved_context.csv')['text'].values.tolist()
  data = DataGenerator.generate([(t,'none') for t in texts], {})

  ResponseGenerator.generate("results/llama3_seed.csv", data, prompts,
                               lambda prompt, text: ollama.chat(model='llama3:instruct', messages=[
        { 'role': 'system', 'content': prompt, },
        { 'role': 'user', 'content': text, },
      ])['message']['content'])
