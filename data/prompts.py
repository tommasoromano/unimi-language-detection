import pandas as pd
import ollama

PROMPTS_JOB_V0 = {
  'zsl': [
    """Sei un assistente che legge ed analizza delle frasi di annunci di lavoro in italiano. Il tuo obbiettivo è analizzare ed identificare se il testo contiene nomi di professioni e se la frase si sta riferendo ad entrambi i generi. Rispondi solamente con il label "NON INCLUSIVO" o "INCLUSIVO".""",
  ],
  'fsl': [
    '''Sei un assistente che legge ed analizza delle frasi di annunci di lavoro in italiano. Il tuo obbiettivo è analizzare ed identificare se il testo contiene nomi di professioni e se la frase si sta riferendo ad entrambi i generi. Rispondi solamente con il label "NON INCLUSIVO" o "INCLUSIVO". Esempi:
    Frase: Requisiti:
    Risposta: INCLUSIVO
    
    Frase: Studio Legale Rossi & Associati cerca programmatrice
    Risposta: NON INCLUSIVO
    
    Frase: Archiviazione e gestione della documentazione fisica e digitale
    Risposta: INCLUSIVO
    
    Frase: Cerchiamo un programmatore
    Risposta: NON INCLUSIVO
    
    Frase: Cerchiamo un programmatore/programmatrice
    Risposta: INCLUSIVO
    
    Frase: Offriamo:
    Risposta: INCLUSIVO
    
    Rispondi solamente con il label "NON INCLUSIVO" o "INCLUSIVO"''',
  ],
  'cot': [
    '''Frase: Studio Legale Rossi & Associati cerca programmatrice
    Risposta: la frase si riferisce a programmatrice che è solo femminile quindi la rispsta è NON INCLUSIVO

    Frase: Requisiti:
    Risposta: la frase non si riferisce ad un particolare genere, quindi la risposta è INCLUSIVO
    
    Frase: Archiviazione e gestione della documentazione fisica e digitale
    Risposta: la frase non si riferisce ad un particolare genere, quindi la risposta è INCLUSIVO
    
    Frase: Cerchiamo un psicologo
    Risposta: la frase si riferisce a psicologo che è solo maschile quindi la risposta è NON INCLUSIVO
    
    Frase: Cerchiamo un cuoco/cuoca
    Risposta: la frase si riferisce ad entrambi i generi quindi la risposta è INCLUSIVO
    
    Frase: Offriamo:
    Risposta: la frase non si riferisce ad un particolare genere, quindi la risposta è INCLUSIVO
    
    Frase: [FRASE]
    Risposta:''',
    '''Sei un assistente che legge ed analizza delle frasi di annunci di lavoro in italiano. Il tuo obbiettivo è analizzare ed identificare se il testo contiene nomi di professioni e se la frase si sta riferendo ad entrambi i generi. Rispondi solamente con il label "NON INCLUSIVO" o "INCLUSIVO".
    Frase: Studio Legale Rossi & Associati cerca programmatrice
    Risposta: la frase si riferisce a programmatrice che è solo femminile quindi la rispsta è NON INCLUSIVO

    Sei un assistente che legge ed analizza delle frasi di annunci di lavoro in italiano. Il tuo obbiettivo è analizzare ed identificare se il testo contiene nomi di professioni e se la frase si sta riferendo ad entrambi i generi. Rispondi solamente con il label "NON INCLUSIVO" o "INCLUSIVO".
    Frase: Requisiti:
    Risposta: la frase non si riferisce ad un particolare genere, quindi la risposta è INCLUSIVO
    
    Sei un assistente che legge ed analizza delle frasi di annunci di lavoro in italiano. Il tuo obbiettivo è analizzare ed identificare se il testo contiene nomi di professioni e se la frase si sta riferendo ad entrambi i generi. Rispondi solamente con il label "NON INCLUSIVO" o "INCLUSIVO".
    Frase: Archiviazione e gestione della documentazione fisica e digitale
    Risposta: la frase non si riferisce ad un particolare genere, quindi la risposta è INCLUSIVO
    
    Sei un assistente che legge ed analizza delle frasi di annunci di lavoro in italiano. Il tuo obbiettivo è analizzare ed identificare se il testo contiene nomi di professioni e se la frase si sta riferendo ad entrambi i generi. Rispondi solamente con il label "NON INCLUSIVO" o "INCLUSIVO".
    Frase: Cerchiamo un psicologo
    Risposta: la frase si riferisce a psicologo che è solo maschile quindi la risposta è NON INCLUSIVO
    
    Sei un assistente che legge ed analizza delle frasi di annunci di lavoro in italiano. Il tuo obbiettivo è analizzare ed identificare se il testo contiene nomi di professioni e se la frase si sta riferendo ad entrambi i generi. Rispondi solamente con il label "NON INCLUSIVO" o "INCLUSIVO".
    Frase: Cerchiamo un cuoco/cuoca
    Risposta: la frase si riferisce ad entrambi i generi quindi la risposta è INCLUSIVO
    
    Sei un assistente che legge ed analizza delle frasi di annunci di lavoro in italiano. Il tuo obbiettivo è analizzare ed identificare se il testo contiene nomi di professioni e se la frase si sta riferendo ad entrambi i generi. Rispondi solamente con il label "NON INCLUSIVO" o "INCLUSIVO".
    Frase: Offriamo:
    Risposta: la frase non si riferisce ad un particolare genere, quindi la risposta è INCLUSIVO
    
    Sei un assistente che legge ed analizza delle frasi di annunci di lavoro in italiano. Il tuo obbiettivo è analizzare ed identificare se il testo contiene nomi di professioni e se la frase si sta riferendo ad entrambi i generi. Rispondi solamente con il label "NON INCLUSIVO" o "INCLUSIVO".
    Frase: [FRASE]
    Risposta:''',
  ],
  'c': [
    'Sei un assistente che legge ed analizza un testo italiano.',
  ],
  'i': [
      'Il tuo obbiettivo è identificare se il testo ed in particolare se i nomi di professioni si riferiscono a maschile o femminile o neutro.',
  ],
  'e': [
      'Ad esempio, impiegat*, impiegato/a, e impiegatÃ sono tutti esempi di parole neutre.',
      'Per esempio, parole che contengono *, /, o Ã sono parole neutre.',
  ],
  'q': [
      'Questa è la domanda: "Il testo si riferisce a maschile o femminile o neutro?"',
  ],
  't': [
      'Puoi rispondere solamente con un label "maschile", "femminile" o "neutro".',
  ]
}

PROMPTS_JOB_V0_KEYS = [['zsl',],['fsl',],['cot',],]
PROMPTS_JOB_V0_KEYS_FINETUNING = [['zsl',],]
def PROMPTS_JOB_V0_F(prompt, text, model):
  messages = [
    { 'role': 'system', 'content': prompt, },
    { 'role': 'user', 'content': text }
  ]
  if prompt in PROMPTS_JOB_V0['cot']:
    messages = [
      { 'role': 'user', 'content': prompt.replace('[FRASE]',text), },
    ]
  # print(messages)
  res = ollama.chat(model=model, messages=messages)['message']['content']
  return res


# pts = pd.read_csv('data/prompts.csv')

# PROMPTS_JOB_CIEQT_V1 = {
#     'c': pts['C'].dropna().unique().tolist(),
#     'i': pts['I'].dropna().unique().tolist(),
#     'e': pts['E'].dropna().unique().tolist(),
#     'q': pts['Q'].dropna().unique().tolist(),
#     't': pts['T'].dropna().unique().tolist(),
# }


################################ OLD PROMPTS ################################

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