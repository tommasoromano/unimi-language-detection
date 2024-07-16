import pandas as pd
import ollama

PROMPTS_JOB_SPLIT_V0 = {
  'zsl': [
    # """Sei un assistente che legge ed analizza delle frasi di annunci di lavoro in italiano. Il tuo obbiettivo è analizzare ed identificare se il testo contiene nomi di professioni o parole inclusive per entrambi i generi. La frase può essere anche generica e non rilevante. Rispondi solamente con il label "NON INCLUSIVO" o "INCLUSIVO" o "NON RILEVANTE".""",
    """Sei un assistente che legge ed analizza delle frasi di annunci di lavoro in italiano. Il tuo obbiettivo è analizzare ed identificare se il testo contiene nomi di professioni e se la frase si sta riferendo ad entrambi i generi. Rispondi solamente con il label "NON INCLUSIVO" o "INCLUSIVO".""",
#     """Sei un assistente che legge ed analizza delle frasi di annunci di lavoro in italiano. Il tuo obbiettivo è analizzare ed identificare se il testo contiene nomi di professioni o parole inclusive per entrambi i generi. La frase può essere anche generica e non rilevante. Rispondi solamente con il label "NON INCLUSIVO" o "INCLUSIVO" o "NON RILEVANTE".

# Frase: Requisiti:
# Q: Rispondi con il label "NON INCLUSIVO" o "INCLUSIVO" o "NON RILEVANTE".
# A: NON RILEVANTE

# Frase: Studio Legale Rossi & Associati cerca programmatrice Legale
# Q: Rispondi con il label "NON INCLUSIVO" o "INCLUSIVO" o "NON RILEVANTE".
# A: NON INCLUSIVO

# Frase: Archiviazione e gestione della documentazione fisica e digitale
# Q: Rispondi con il label "NON INCLUSIVO" o "INCLUSIVO" o "NON RILEVANTE".
# A: NON RILEVANTE

# Frase: programmatrice svolgerà un ruolo chiave nel supportare le attività quotidiane dello studio, 
# Q: Rispondi con il label "NON INCLUSIVO" o "INCLUSIVO" o "NON RILEVANTE".
# A: NON INCLUSIVO

# Frase: garantendo un'efficiente gestione amministrativa e contribuendo al successo complessivo del team legale.
# Q: Rispondi con il label "NON INCLUSIVO" o "INCLUSIVO" o "NON RILEVANTE".
# A: NON RILEVANTE

# Frase: Organizzazione di viaggi e trasferte per i professionisti dello studio
# Q: Rispondi con il label "NON INCLUSIVO" o "INCLUSIVO" o "NON RILEVANTE".
# A: NON INCLUSIVO"""
  ]
}
PROMPTS_JOB_SPLIT_V0_KEYS = [['zsl',],]
def PROMPTS_JOB_SPLIT_V0_F(prompt, text, model):
  messages = [
    { 'role': 'system', 'content': prompt, },
    { 'role': 'user', 'content': text }
  ]
  # print(messages)
  res = ollama.chat(model=model, messages=messages)['message']['content']
  return res

# def PROMPTS_JOB_SPLIT_V0_F(prompt, text, model):
#   lines = text.split('\n')
#   for line in lines:
#     line = line.replace('\n','')
#     if line == '': continue
#     # messages = [
#     #   { 'role': p.split(' ')[0].replace(':','').lower(), 
#     #    'content': p.replace('System: ','').replace('User: ','').replace('Assistant: ',''), }
#     #   for p in prompt.split('\n')
#     #   ]+[
#     #   { 'role': 'user', 'content': text, }
#     # ]
#     messages = [
#       { 'role': 'system', 'content': prompt, },
#       { 'role': 'user', 'content': line }
#     ]
#     # print(messages)
#     res = ollama.chat(model=model, messages=messages)['message']['content']
#     print(res, line)
#   return "split"

# def PROMPTS_JOB_SPLIT_V0_F(prompt, text, model):
#   lines = text.split('\n')
#   for line in lines:
#     res = ollama.chat(model=model, messages=[
#       { 'role': 'system', 'content': prompt, },
#       { 'role': 'user', 'content': text, }
#     ])['message']['content']
#     print(res, line)
#   return "split"

PROMPTS_COT_V0 = {
    'cot': [
        """Context: Sei un assistente che legge ed analizza un testo italiano. Il tuo obbiettivo è identificare se il testo è inclusivo verso tutti i generi. La tua risposta deve essere nella forma: INCLUSIVO o "NON INCLUSIVO, parola o frase target, spiegazione.
        Input: Sei pronto a #makeanimpactthatmatters nel nostro team Operating Model Transformation FSI?
        Risposta: NON INCLUSIVO, Sei pronto, Il testo è non inclusivo perché Sei pronto si riferisce solo al genere maschile.
        Input: Hai la capacità di trasformare un problema in opportunità Hai la capacità di trasformare un problema in opportunità. Sei empatica e in grado di cogliere i need dei tuoi colleghi e degli stakeholder interni ed esterni. Ti piace lavorare in un ambiente sfidante in cui le priorità possono cambiare quotidianamente. Ti piacerebbe imparare ad avere una visione progettuale d'insieme, tenendo sempre a mente l'operatività per raggiungere gli obiettivi con integrità e rispetto degli standard Deloitte. Vuoi puntare sulla crescita continua delle tue competenze per condividerle con tutto il team.
        Risposta: NON INCLUSIVO, Sei empatica, Il testo è non inclusivo perché Sei empatica si riferisce solo al genere femminile.
        Input: In quanto Consultant/Senior Consultant avrai la possibilità di condividere la tua esperienza e le tue conoscenze con i colleghi più giovani, iniziando a sviluppare capacità di leadership, e individuando la migliore soluzione per il cliente, occupandoti di: "Raccogliere e analizzare i dati necessari all'elaborazione di soluzioni per i clienti; Raccogliere e analizzare i dati necessari all'elaborazione di soluzioni per i clienti;  Collaborare con il team di progetto in modo proattivo e collaborativo; Comunicare in modo chiaro e strutturato messaggi e contenuti utili alla predisposizione di deliverable progettuali;
        Risposta: INCLUSIVO."""
    ]
}
PROMPTS_COT_V0_KEYS = [['cot',],]
def __f(prompt, text):
  def f(ln):
    if ln.startswith('Context: '):
      return { 'role': 'system', 'content': ln.split('Context: ')[1], }
    if ln.startswith('Input: '):
      return { 'role': 'user', 'content': ln.split('Input: ')[1], }
    if ln.startswith('Risposta: '):
      return { 'role': 'assistant', 'content': ln.split('Risposta: ')[1], }
    return { 'role': 'user', 'content': text, }
  messages = [f(ln) for ln in prompt.split('\n')]
  return messages+[f(text)]
PROMPTS_COT_V0_OLLAMA = __f

PROMPTS_JOB_CIEQT_V0 = {
    'c': [
        'Sei un assistente che legge ed analizza un testo italiano.',
    ],
    'i': [
        'Il tuo obbiettivo è identificare se il testo ed in particolare se i nomi di professioni si riferiscono a maschile o femminile o neutro.',
        'Il tuo obbiettivo è di identificare se il testo si riferisce a maschile o femminile o neutro (entrambi).',
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
        'Classifica il testo in base al genere maschile o femminile o neutro.',
        'Rispondi tenendo la struttura "genere" + "spiegazione"'
        'Classifica il testo in base al genere maschile o femminile o neutro con struttura "genere" + "spiegazione".',
    ]
}

pts = pd.read_csv('data/prompts.csv')

PROMPTS_JOB_CIEQT_V1 = {
    'c': pts['C'].dropna().unique().tolist(),
    'i': pts['I'].dropna().unique().tolist(),
    'e': pts['E'].dropna().unique().tolist(),
    'q': pts['Q'].dropna().unique().tolist(),
    't': pts['T'].dropna().unique().tolist(),
}


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