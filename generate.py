import ollama
from openai import OpenAI
import pandas as pd
import os
from nlp_synt_data import *
from data.prompts import *
from data.texts import *
from sc import *

# ollama run llama3:instruct
# ollama run mistral:instruct
# ollama run gemma:instruct
# ollama run phi3:instruct

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

  SEED = True

  # Generate synthetic data and run models
  prompts = PromptGenerator.generate(
    PROMPTS_JOB_V0, 
    PROMPTS_JOB_V0_KEYS,
    # [['zsl',],],
    )

  # data = DataGenerator.generate(TEXT_v0, SUBS_JOBS_V0)
  if SEED:
    data = DataGenerator.generate(
      TEXT_SEED_v0(), 
      SUBS_JOBS_V0)
  else:
    data = sc_make_test_data()

  # model = 'llama3:instruct'
  n_pass = 1

  client = OpenAI(
      # This is the default and can be omitted
      api_key="",
  )

  # df_fix_text = pd.read_csv(sc_csv_name('gpt-4o-minii', False, SEED))

  for model in [
    # 'llama3.1',
    # 'llama3:instruct',
    # 'mistral',
    # 'gemma2',
    # 'qwen2',
    # 'phi3:mini',
    "gpt-4o-mini",
    ]:
    print(model)
    # model_f = lambda messages: ollama.chat(model=model, messages=messages)['message']['content']
    model_f = lambda messages: client.chat.completions.create(
      model=model,
      messages=messages
    ).choices[0].message.content

    # def fn(prompt, text):
    #   idx = [i for i,d in enumerate(sc_make_test_data()) if d.text == text][0]
    #   response = df_fix_text.iloc[idx]['response']
    #   return response
    ResponseGenerator.generate(sc_csv_name(model, False, SEED), 
                               data, 
                               prompts,
                               lambda prompt, text: PROMPTS_JOB_V0_F(prompt, text, model_f), 
                              #  fn,
                               n_pass=n_pass,
                               )
