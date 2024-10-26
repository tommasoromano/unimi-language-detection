import pandas as pd
from .jobs import *

JOBS_SPLIT_v0_n0 = [
('Studio Legale Rossi & Associati cerca [JOB] Legale',"TODO"),
('Lo Studio Legale Rossi & Associati, rinomato studio con sede nel centro di Milano specializzato in diritto societario e commerciale, è alla ricerca di una [JOB] Legale per unirsi al nostro team dinamico e in crescita.',"TODO"),
('Descrizione del ruolo:',"INCLUSIVO"),
("[JOB] svolgerà un ruolo chiave nel supportare le attività quotidiane dello studio, garantendo un'efficiente gestione amministrativa e contribuendo al successo complessivo del team legale.","TODO"),
('Responsabilità principali:',"INCLUSIVO"),
# ("Gestione dell'agenda e organizzazione degli appuntamenti per gli avvocati dello studio","NON INCLUSIVO"),
('Accoglienza dei clienti e gestione delle comunicazioni telefoniche ed email',"INCLUSIVO"),
('Preparazione e formattazione di documenti legali, lettere e relazioni',"INCLUSIVO"),
('Archiviazione e gestione della documentazione fisica e digitale',"INCLUSIVO"),
('Supporto nella preparazione di fascicoli per udienze e riunioni',"INCLUSIVO"),
# ('Coordinamento con altri professionisti e studi legali',"NON INCLUSIVO"),
('Gestione delle pratiche amministrative e contabili di base',"INCLUSIVO"),
# ('Organizzazione di viaggi e trasferte per i professionisti dello studio',"NON INCLUSIVO"),
('Requisiti:',"INCLUSIVO"),
('Diploma di scuola superiore, preferibilmente in ambito amministrativo o giuridico',"INCLUSIVO"),
('Esperienza pregressa di almeno 2-3 anni in ruolo analogo, preferibilmente in ambito legale',"INCLUSIVO"),
('Ottima conoscenza del pacchetto Office e dei principali software gestionali',"INCLUSIVO"),
('Eccellenti capacità organizzative e di gestione del tempo',"INCLUSIVO"),
('Ottime doti comunicative e relazionali',"INCLUSIVO"),
('Capacità di lavorare in modo autonomo e in team',"INCLUSIVO"),
('Discrezione e riservatezza nella gestione delle informazioni sensibili',"INCLUSIVO"),
('Flessibilità e capacità di gestire lo stress',"INCLUSIVO"),
('Buona conoscenza della lingua inglese (scritta e parlata)',"INCLUSIVO"),
('Offriamo:',"INCLUSIVO"),
('Contratto a tempo indeterminato con pacchetto retributivo competitivo',"INCLUSIVO"),
('Ambiente di lavoro stimolante e professionale',"INCLUSIVO"),
('Possibilità di crescita e sviluppo professionale',"INCLUSIVO"),
('Formazione continua e aggiornamenti nel settore legale',"INCLUSIVO"),
('Benefits aziendali (assicurazione sanitaria integrativa, buoni pasto)',"INCLUSIVO"),
('Se sei [JOB] per contribuire al successo di uno studio legale in crescita, inviaci il tuo CV e una lettera di presentazione. Valuteremo attentamente tutte le candidature e contatteremo i profili più in linea con le nostre esigenze per un colloquio.',"TODO"),
('Lo Studio Legale Rossi & Associati è un datore di lavoro che offre pari opportunità e valorizza la diversità nel luogo di lavoro.',"INCLUSIVO"),
('Per candidarti, invia il tuo CV all\'indirizzo email: recruiting@rossiassociati.it con oggetto "Candidatura Segretaria Legale - [Nome e Cognome]"',"INCLUSIVO"),
]

JOBS_SPLIT_v0_n1 = [
('Siamo ansiosi di conoscerti e di esplorare come le tue competenze e la tua esperienza possano contribuire al nostro team!',"INCLUSIVO"),
("Siamo alla ricerca di [JOB] per una prestigiosa realtà del settore bancario con sede in provincia di Bergamo. ","TODO"),
("Requisiti richiesti: ","INCLUSIVO"),
("- Diploma di scuola superiore e/o laurea in discipline economiche o finanziarie ","INCLUSIVO"),
("- Esperienza pregressa nel ruolo di [JOB] allo sportello bancario ","INCLUSIVO"),
("- Capacità di gestire le operazioni di cassa, inclusi pagamenti, prelievi e depositi ","INCLUSIVO"),
("- Conoscenza dei principali strumenti informatici e software bancari ","INCLUSIVO"),
("- Capacità di lavorare in autonomia e di gestire situazioni di stress ","INCLUSIVO"),
("- Ottime capacità comunicative e relazionali ","INCLUSIVO"),
("- Precisione e attenzione ai dettagli Responsabilità: ","INCLUSIVO"),
("- Gestione delle operazioni di cassa, inclusi pagamenti, prelievi e depositi ","INCLUSIVO"),
("- Accoglienza e assistenza ai clienti presso lo sportello bancario ","INCLUSIVO"),
("- Fornire informazioni sui prodotti e servizi bancari ","INCLUSIVO"),
("- Promuovere e vendere prodotti e servizi finanziari ","INCLUSIVO"),
("- Mantenere la cassa in ordine e bilanciata ","INCLUSIVO"),
("- Collaborare con il team per raggiungere gli obiettivi di vendita ","INCLUSIVO"),
("Offriamo: ","INCLUSIVO"),
("- Contratto di assunzione diretta ","INCLUSIVO"),
("- Orario di lavoro a giornata ","INCLUSIVO"),
("- Ambiente di lavoro stimolante e dinamico ","INCLUSIVO"),
("- Possibilità di crescita professionale ","INCLUSIVO"),
("Se sei una persona motivata, con esperienza nel settore bancario e con ottime capacità di gestione della cassa e dello sportello, inviaci il tuo curriculum vitae. ","INCLUSIVO"),
("Sarai contattato/a per un colloquio conoscitivo. ","INCLUSIVO"),
# ("Sarai contattato per un colloquio conoscitivo. ","NON INCLUSIVO"),
# ("Sarai contattata per un colloquio conoscitivo. ","NON INCLUSIVO"),
("Si prega di inviare il proprio curriculum vitae indicando nell'oggetto del messaggio il titolo della posizione per cui si candida.","INCLUSIVO"),
("Grazie per l'interesse dimostrato. ","INCLUSIVO"),
# ("I candidati, nel rispetto del D.lgs. 198/2006, D.lgs 215/2003 e D.lgs 216/2003, sono invitati a leggere l'informativa sulla privacy (Regolamento UE n. 2016/679).","NON INCLUSIVO"),
]

JOBS_SPLIT_v0_n2 = [
("Hai spirito di iniziativa? Hai passione per la ristorazione e hai maturato esperienza come [JOB]?","TODO"),
("Candidati subito a questa opportunità!","INCLUSIVO"),
("OPPORTUNITÀ","INCLUSIVO"),
("Per una strutturata realtà nel settore della ristorazione, il Team di Adecco Explora è alla ricerca di [JOB] a Cremona.","TODO"),
("RESPONSABILITÀ","INCLUSIVO"),
("-Gestione e attenzione al cliente;","INCLUSIVO"),
("-Presa comanda con palmare;","INCLUSIVO"),
("-Apparecchiare e sparecchiare i tavoli;","INCLUSIVO"),
("-Runner piatti e bibite;","INCLUSIVO"),
("-Gestione del bar e della cassa.","INCLUSIVO"),
("Responsabilità:","INCLUSIVO"),
("REQUISITI","INCLUSIVO"),
("- Esperienza, anche minima, nel ruolo e nel servizio al tavolo;","INCLUSIVO"),
("- Capacità di lavorare in team;","INCLUSIVO"),
("- Predisposizione al contatto con il cliente.","INCLUSIVO"),
("ORARIO DI LAVORO:","INCLUSIVO"),
("Part Time, 20 ore settimanali, dal lunedì alla domenica con un giorno di riposo su turnazione.","INCLUSIVO"),
("Si richiede disponibilità a turni spezzati.","INCLUSIVO"),
("Inserimento iniziale a tempo determinato, scopo assuntivo. CCNL Turismo Pubblici esercizi VI Livello da commisurare in base all'esperienza.","INCLUSIVO"),
]

JOBS_SPLIT_v0_n3 = [
("tnservice srl, agenzia per il lavoro (aut. min r. 0000157 del 09/11/2022) è specializzata nella selezione del personale.","INCLUSIVO"),
("per nostro cliente specializzato nel settore turismo e viaggi stiamo cercando [JOB]","TODO"),
("[JOB], con indispensabile esperienza pregressa presso agenzia viaggi, verrà inserita in ufficio front client in qualità di agente business travels.","TODO"),
("la risorsa, con indispensabile esperienza pregressa presso agenzia viaggi, verrà inserita in ufficio front client in qualità di [JOB].","TODO"),
("mansioni:","INCLUSIVO"),
("organizzazione in completa autonomia (senza ausilio di programmi tour operator) di viaggi business per clienti.","INCLUSIVO"),
("gestione delle prenotazioni tramite il sistema trenitalia pico","INCLUSIVO"),
("gestione amministrativa contabile delle fatture - intrastat ecc.","INCLUSIVO"),
("requisiti:","INCLUSIVO"),
("utilizzo del sistema pico.","INCLUSIVO"),
("utilizzo del sistema gds sabre.","INCLUSIVO"),
("ottima competenza nel programma zucchetti.","INCLUSIVO"),
("conoscenza dei sistemi web, microsoft e sistemi di prenotazione online.","INCLUSIVO"),
("abilità e flessibilità mentale per la creazione di pacchetti viaggio.","INCLUSIVO"),
("conoscenza delle lingue straniere.","INCLUSIVO"),
("competenze trasversali:","INCLUSIVO"),
("serietà, affidabilità... precisione e predisposizione al contatto con il pubblico.","INCLUSIVO"),
("inserimento: diretto in azienda a tempo indeterminato.","INCLUSIVO"),
("retribuzione: commisurata in base alle competenze della risorsa.","INCLUSIVO"),
("orario: full time da lunedì al venerdì, dalle 9 alle 12.30 e dalle 14.30 alle 19.00","INCLUSIVO"),
("luogo di lavoro: cassano dadda (mi)","INCLUSIVO"),
("il presente annuncio è rivolto ad entrambi i sessi, ai sensi delle leggi 903/77 e 125/91 e a persone di tutte le età e tutte le nazionalità, ai sensi dei decreti legislativi 215/03 e 216/03.","INCLUSIVO"),
("ti chiediamo di inserire nel cv l'autorizzazione al trattamento dei tuoi dati personali ai sensi del regolamento ue n. 679/2016 e della legislazione italiana vigente.","INCLUSIVO"),
]

JOBS_SPLIT_v0_n4 = [
("sei [JOB] con la passione di creare esperienze di intrattenimento indimenticabili? ","TODO"),
("ti piace esibirti davanti a grandi folle, assicurando loro un'esperienza memorabile? ","INCLUSIVO"),
("stai cercando un'opportunità per mettere in pratica le tue capacità artistiche, sviluppare nuove abilità, imparare nuovi spettacoli e acquisire una vera esperienza sul palco? ","INCLUSIVO"),
("se è così, abbiamo l’occasione perfetta per te! ","INCLUSIVO"),
("stiamo cercando [JOB] che si unisca al nostro team in grecia per la stagione estiva. ","TODO"),
("con una retribuzione competitiva, un'assicurazione sanitaria e professionale completa, vitto e alloggio gratuiti e trasporto bonus, questa è un'opportunità da non perdere! ","INCLUSIVO"),
("candidati ora con il tuo cv e la lettera di presentazione per unirti al nostro team e rendere questa estate indimenticabile. ","INCLUSIVO"),
("località: grecia o cipro ","INCLUSIVO"),
("tipologia di lavoro: stagionale, estivo ","INCLUSIVO"),
("lingua: inglese (+ altre... lingue dell'ue preferite) ","INCLUSIVO"),
("stipendio: stipendio mensile competitivo (basato su esperienza e competenze) ","INCLUSIVO"),
("periodo: 6 - 7 mesi, stagione estiva 2024 ","INCLUSIVO"),
("benefici: • assicurazione sanitaria e professionale completa, • alloggio fornito, • pasti forniti in pensione completa e • una tariffa di trasporto bonus! ","INCLUSIVO"),
]

JOBS_SEED_SPLIT_v0_n0 = [
("#YourRole Il tuo ruolo, in quanto [JOB], ti darà la possibilità di concretizzare in prima persona iniziative per progetti strategici, condividendo la tua esperienza e le tue conoscenze con colleghi di talento, iniziando a sviluppare capacità di leadership, e individuando la migliore soluzione per i clienti, occupandoti di: ","TODO"),
("Gestione e coordinamento delle risorse più junior, in termini di definizione delle tempistiche, utilizzo di specifici tool, controllo di qualità della delivery; ","INCLUSIVO"),
("Definizione delle metodologie ed approcci innovativi per la pianificazione e il monitoraggio delle iniziative in termini di Project Management e/o Program Management; ","INCLUSIVO"),
("Realizzazione e monitoraggio delle attività attraverso SAL periodici, sia interni che esterni, con i principali stakeholder; ","INCLUSIVO"),
("Definizione di attività di analisi del flusso documentale di progetto in termini di data entry e archiviazione; ","INCLUSIVO"),
("Gestione e controllo della qualità della delivery progettuale; ","INCLUSIVO"),
("Organizzazione delle attività di reportistica e analisi dei KPI in termini di Risk Management e Quality Management.","INCLUSIVO"),
]

JOBS_SEED_SPLIT_v0_n1 = [
("Il profilo ideale è [JOB] che: ","TODO"),
("Ha conseguito una laurea magistrale in ingegneria gestionale o in ambito economico/manageriale ","INCLUSIVO"),
("Ha un’esperienza pregressa maturata in società di consulenza strategico/organizzativa di almeno 3 anni, in modo particolare su clienti del mercato Retail e Fast Moving Consumer Goods ","INCLUSIVO"),
# ("È stato coinvolto in progettualità di redazioni di piani industriali e strategici ","NON INCLUSIVO"),
# ("È stata coinvolta in progettualità di redazioni di piani industriali e strategici ","NON INCLUSIVO"),
("Possiede comprovata esperienza (almeno 3 anni) in progetti di Due Diligence e Data Strategy ","INCLUSIVO"),
("Possiede comprovata esperienza (almeno 3 anni) su tematiche afferenti all’analisi dei processi aziendali ","INCLUSIVO"),
("Possiede comprovata esperienza nel relazionarsi in modo autonomo direttamente con i referenti lato cliente, interagendo con i C-Level tanto nelle fasi di ricerca quanto nell’acquisizione delle informazioni necessarie per il disegno del progetto e la gestione dei SAL ","INCLUSIVO"),
("Ha esperienza di almeno 2 anni di gestione, coordinamento e sviluppo di risorse junior ","INCLUSIVO"),
("Ha ottima conoscenza/competenze con pacchetto office, in modo particolare Excel e PPT ","INCLUSIVO"),
("Ha ottima conoscenza della lingua inglese","INCLUSIVO"),
]

JOBS_SEED_SPLIT_v0_n2 = [
("[JOB] si occuperà delle seguenti attività: ","TODO"),
("Gestione e smistamento corrispondenza generale: mail, telefono, chat; ","INCLUSIVO"),
("Gestione sale riunioni: prenotazioni e cura dell’ordine delle sale; ","INCLUSIVO"),
("Gestione attività di segreteria generale: a supporto di colleghi e responsabili; ","INCLUSIVO"),
("Monitoraggio sistemi di sicurezza; ","INCLUSIVO"),
("Gestione trasferte e viaggi; ","INCLUSIVO"),
("Coordinamento eventi interni: riunioni interne, meeting, formazione, piccoli eventi interni; ","INCLUSIVO"),
("Attività varie a supporto dell'area Marketing.","INCLUSIVO"),
]

JOBS_SEED_SPLIT_v0_n3 = [
("In base al grade di [JOB] avrai la possibilità di condividere la tua esperienza e le tue conoscenze con il team, iniziando a sviluppare capacità di leadership, e individuando la migliore soluzione occupandoti di: ","TODO"),
("Interpretare proattivamente i bisogni del cliente, individuando metodologie e approcci coerenti ","INCLUSIVO"),
("Sviluppare e gestire iniziative e soluzioni in modo indipendente,  riorganizzazione aziendale o di ristrutturazione del debito finanziario.  ","INCLUSIVO"),
("Sviluppare iniziative e soluzioni in modo indipendente, raccogliendo e strutturando dati e svolgendo analisi a supporto delle decisioni di business.  ","INCLUSIVO"),
("Valorizzare il contributo delle risorse più Junior e contribuire alla loro crescita. ","INCLUSIVO"),
("Individuare potenziali rischi e criticità e condividerli tempestivamente con il resto del team. ","INCLUSIVO"),
("Collaborare efficacemente in team eterogenei ed internazionali. ","INCLUSIVO"),
("Grazie all’ufficio Learning avrai modo di seguire corsi e attività di formazione che ti permetteranno di ampliare e approfondire le tue conoscenze. ","INCLUSIVO"),
("Il team del Talent invece, si occuperà in maniera strutturata del tuo avanzamento di carriera con KPI e obiettivi definiti, che se raggiunti, ti garantiranno la revisione del grade e degli economics.","INCLUSIVO"),
]

JOBS_SEED_SPLIT_v0_n4 = [
("Entrando a far parte del team in qualità di [JOB], potrai occuparti di: ","TODO"),
("Gaap Conversion & Finance Transformation : implementazione di nuovi principi contabili nonché disegno del modello contabile e di processo, definizione degli impatti di business ed organizzativi ed individuazione dei requisiti target a supporto della implementazione dei nuovi principi contabili; ","INCLUSIVO"),
("ESG assistance : supporto nel percorso di predisposizione di bilanci di sostenibilità e progetti di adeguamento alla normativa Corporate Sustainability Reporting Directive (CSRD);  ","INCLUSIVO"),
("Planning and Control : revisione del modello e dei processi di pianificazione e controllo anche ai fini della compliance alle normative di riferimento per le società quotate in Borsa e per gli enti di interesse pubblico; ","INCLUSIVO"),
("Technologies change e Smart Digital Finance : sviluppo di soluzioni/piattaforme e processi di automation anche mediante l’uso di nuove tecnologie come Intelligenza Artificiale e Machine Learning ; ","INCLUSIVO"),
("Change Management : supporto nella definizione dei nuovi ruoli e responsabilità all'interno dell'organizzazione, alla luce delle progettualità in essere; ","INCLUSIVO"),
("individuazione delle principali novità operative e gestionali; ","INCLUSIVO"),
("analisi dei principali fattori critici alla base del cambiamento;  ","INCLUSIVO"),
("Project Management : predisposizione e monitoraggio del piano delle attività, coordinamento tra gli Stakeholder di Business e IT durante le fasi implementative, predisposizione documentazione progettuale per Top Management; ","INCLUSIVO"),
("Accounting advice in ambito Deal : supporto nella predisposizione del reporting finanziario richiesto nell’ambito di progetti di transaction services o di operazioni straordinarie. ","INCLUSIVO"),
]

ADJS_SPLIT_v0_n0 = [
("sei giovane, [ADJ] del settore immobiliare? ","TODO",),
("[ADJ] a raggiungere gli obiettivi ","TODO",),
("hai un’esperienza contabile o sei già amministratore di condominio, [ADJ] con solide competenze amministrative ed uno spirito positivo e che vuole lavorare in team?","TODO",),
("siamo alla ricerca di una figura [ADJ] per unirsi al nostro team presso la farmacia lunghi situata a corteolona. ","TODO"),
("[ADJ] ad inviare il proprio cv, con autorizzazione ai sensi del Regolamento (UE). 2016/679 e del D.Lgs., 196/2003 s.m.i.", "TODO"),
("Sei [ADJ] a #makeanimpactthatmatters nel nostro team di Deloitte Officine Innovazione?","TODO"),
("se sei [ADJ] a mettere al servizio del nostro studio la tua esperienza e le tue competenze contabili, inviaci il tuo curriculum e una breve presentazione motivazionale.","TODO"),
]

ADJS_SPLIT_v0_n1 = [
("se sei [ADJ] a iniziare la tua carriera nel mondo immobiliare, inviaci il tuo curriculum vitae all'indirizzo e-mail info@tirreniaimmobiliare.com o fissa un appuntamento telefonando allo 090/9486682.","TODO"),
("se sei [ADJ] a far parte della nostra squadra dedicata e professionale, inviaci il tuo curriculum vitae oppure contattami su whatsapp al numero 3385906783","TODO"),
("se sei [ADJ] al cliente e di contribuire al benessere della comunità, potresti essere la persona giusta per noi.","TODO"),
]

JOBS_OTHER_SPLIT_v0_n0 = [
("[JOB] può inviare il proprio cv, con autorizzazione ai sensi del Regolamento (UE). 2016/679 e del D.Lgs., 196/2003 s.m.i.", "TODO"),
("[JOB] ideale ha maturato un’esperienza di almeno 5 anni sui moduli MM e PP in contesti SAP fortemente personalizzati e possiede una buona conoscenza del modulo SD.","TODO"),
("Sei [JOB] a #makeanimpactthatmatters nel nostro team di Deloitte Officine Innovazione?","TODO"),
('Coordinamento con [JOB]',"TODO"),
("Gestione dell'agenda e organizzazione degli appuntamenti per [JOB] dello studio","TODO"),
# MISC
("se sei una persona dinamica, orientata al cliente e desiderosa di contribuire al benessere della comunità, potresti essere la persona giusta per noi.","INCLUSIVO"),
# VERBS 
("È [VERB] in progettualità di redazioni di piani industriali e strategici ","TODO"),
("Sarai [VERB] per un colloquio conoscitivo. ","TODO"),
("fitactive, catena leader di mercato per numero di palestre aperte in italia, ricerca [JOB] per inserimento immediato nel team group trainers.","TODO"),
("- abilitazioni riconosciute (anche marchi registrati come zumba, strong nation, cross cardio ecc..)","INCLUSIVO"),
("- grande energia e voglia di crescere professionalmente in una realtà importante e in forte sviluppo nel mondo del fitness","INCLUSIVO"),
("- predisposizione alla formazione e al lavoro di squadra","INCLUSIVO"),
("per candidarti invia il tuo curriculum alla mail indicata, sarai [VERB] il prima possibile!","TODO"),
]

TEXT_JOB_TRAIN_v0 = JOBS_SPLIT_v0_n0 + JOBS_SPLIT_v0_n1 + JOBS_SPLIT_v0_n2 + JOBS_SPLIT_v0_n4 + JOBS_SEED_SPLIT_v0_n0 + JOBS_SEED_SPLIT_v0_n2 + JOBS_SEED_SPLIT_v0_n3 + JOBS_SEED_SPLIT_v0_n4 + ADJS_SPLIT_v0_n0 + JOBS_OTHER_SPLIT_v0_n0
TEXT_JOB_TEST_v0 = JOBS_SPLIT_v0_n3 + JOBS_SEED_SPLIT_v0_n1 + ADJS_SPLIT_v0_n1

TEXT_JOB_ALL_v0 = TEXT_JOB_TRAIN_v0 + TEXT_JOB_TEST_v0

#EXT_v0 = TEXT_JOBS_v0 + TEXT_NON_JOBS_v0

# /content/unimi-language-detection/
# JOBS_V0 = pd.read_csv('data/synt/jobs.csv').values.tolist()
# ADJ
__adj = [
    ("esperto","maschile"),
    ("dinamico","maschile"),
    ("motivato","maschile"),
    ("appassionato","maschile"),
    ("ambizioso","maschile"),
    ("determinato","maschile"),
    ("pronto","maschile"),
    ("interessato","maschile"),
    # ("specializzato","maschile")
    # ("proattivo","maschile"),
    # ("collaborativo","maschile"),
    # ("orientato","maschile"),
    # ("positivo","maschile"),
    # ("attivo","maschile"),
    # ("preciso","maschile"),
]
makeFemale = lambda x: (x[0][:-1]+'a', 'femminile')
makeNeutral = lambda x: (x[0]+'/a', 'neutro')
makeNeutral2 = lambda x: (x[0][:-1]+'*', 'neutro')
__adjf = list(map(makeFemale, __adj))
__adjn = list(map(makeNeutral, __adj))
__adjn += list(map(makeNeutral2, __adj))
__adj = __adj + __adjf + __adjn
# make combinations of them if they have the same 2nd element
__adj2 = [(a[0] + " e " + b[0], a[1]) for a in __adj for b in __adj if a[1] == b[1] and a[0] != b[0]]
__adj3 = [(a[0] + ", " + b[0] + " e " + c[0], a[1]) for a in __adj for b in __adj for c in __adj if a[1] == b[1] and b[1] == c[1] and a[0] != b[0] and b[0] != c[0] and a[0] != c[0]]
ADJ_V0 = __adj + __adj2

__verbm = [
    ("coinvolto","maschile"),
    ("contattato","maschile"),
    ("considerato","maschile"),
    ("chiamato","maschile"),
    ("ricontattato","maschile"),
    ("assunto","maschile"),
]
__verbf = list(map(makeFemale, __verbm))
__verbn = list(map(makeNeutral, __verbm))
__verbn2 = list(map(makeNeutral2, __verbm))
VERBS_v0 = __verbm + __verbf + __verbn + __verbn2

SUBS_JOBS_V0 = {
        "JOB": JOBS_v0,
        "ADJ": ADJ_V0,
        "VERB": VERBS_v0,
        }

######## SEED ########

def TEXT_SEED_v0():
    # /content/unimi-language-detection/
    df = pd.read_csv('data/job_description_seed_dataset_improved_context.csv')
    _map = {
        'YES': 'INCLUSIVO',
        'NO': 'NON INCLUSIVO'
    }
    df['label'] = df.apply(lambda x: _map[x['inclusive phrasing']], axis=1)
    return df[['text','label']].values.tolist()


######### OLD #########

TEXTS_JOBS_FROM_SEED = [
    ("""Il tuo ruolo, in quanto Consultant / Senior Consultant, ti darà la possibilità di concretizzare in prima persona iniziative per progetti strategici, condividendo la tua esperienza e le tue conoscenze con colleghi di talento, iniziando a sviluppare capacità di leadership, e individuando la migliore soluzione per i clienti, occupandoti di: Gestione e coordinamento delle risorse più junior, in termini di definizione delle tempistiche, utilizzo di specifici tool, controllo di qualità della delivery; Definizione delle metodologie ed approcci innovativi per la pianificazione e il monitoraggio delle iniziative in termini di Project Management e/o Program Management; Realizzazione e monitoraggio delle attività attraverso SAL periodici, sia interni che esterni, con i principali stakeholder; Definizione di attività di analisi del flusso documentale di progetto in termini di data entry e archiviazione; Gestione e controllo della qualità della delivery progettuale; Organizzazione delle attività di reportistica e analisi dei KPI in termini di [JOB].""", "end"),
    ("""Il tuo ruolo, in quanto [JOB], ti darà la possibilità di concretizzare in prima persona iniziative per progetti strategici, condividendo la tua esperienza e le tue conoscenze con colleghi di talento, iniziando a sviluppare capacità di leadership, e individuando la migliore soluzione per i clienti, occupandoti di: Gestione e coordinamento delle risorse più junior, in termini di definizione delle tempistiche, utilizzo di specifici tool, controllo di qualità della delivery; Definizione delle metodologie ed approcci innovativi per la pianificazione e il monitoraggio delle iniziative in termini di Project Management e/o Program Management; Realizzazione e monitoraggio delle attività attraverso SAL periodici, sia interni che esterni, con i principali stakeholder; Definizione di attività di analisi del flusso documentale di progetto in termini di data entry e archiviazione; Gestione e controllo della qualità della delivery progettuale; Organizzazione delle attività di reportistica e analisi dei KPI in termini di Risk Management e Quality Management.""", "start"),
    ("""Il tuo ruolo, in quanto Consultant / Senior Consultant, ti darà la possibilità di concretizzare in prima persona iniziative per progetti strategici, condividendo la tua esperienza e le tue conoscenze con colleghi di talento, iniziando a sviluppare capacità di leadership, e individuando la migliore soluzione per i clienti, occupandoti di: Gestione e coordinamento delle risorse più junior, in termini di definizione delle tempistiche, utilizzo di specifici tool, controllo di qualità della delivery; Definizione delle metodologie ed approcci innovativi per la pianificazione e il monitoraggio delle iniziative in termini di [JOB]; Realizzazione e monitoraggio delle attività attraverso SAL periodici, sia interni che esterni, con i principali stakeholder; Definizione di attività di analisi del flusso documentale di progetto in termini di data entry e archiviazione; Gestione e controllo della qualità della delivery progettuale; Organizzazione delle attività di reportistica e analisi dei KPI in termini di Risk Management e Quality Management.""", "middle"),
    ("""Realizzazione e monitoraggio delle attività attraverso SAL periodici, sia interni che esterni, con i principali stakeholder; Definizione di attività di analisi del flusso documentale di progetto in termini di data entry e archiviazione; Gestione e controllo della qualità della delivery progettuale; Organizzazione delle attività di reportistica e analisi dei KPI in termini di [JOB].""", "end"),
    ("""Il tuo ruolo, in quanto [JOB], ti darà la possibilità di concretizzare in prima persona iniziative per progetti strategici, condividendo la tua esperienza e le tue conoscenze con colleghi di talento, iniziando a sviluppare capacità di leadership, e individuando la migliore soluzione per i clienti""", "start"),
    ("""Gestione e coordinamento delle risorse più junior, in termini di definizione delle tempistiche, utilizzo di specifici tool, controllo di qualità della delivery; Definizione delle metodologie ed approcci innovativi per la pianificazione e il monitoraggio delle iniziative in termini di [JOB]; Realizzazione e monitoraggio delle attività attraverso SAL periodici, sia interni che esterni, con i principali stakeholder.""", "middle"),
]

old_texts = """Stiamo cercando [JOB] per la nostra azienda.
Ufficio di nuova aprtura cerca [JOB] con esperienza.
Orario per [JOB]: monte ore variabile da 20 a 36 ore settimanali, in base alla disponibilità.
Manpower filiale di Mondovì è alla ricerca di [JOB] in tutta la provincia di Cuneo.
B-Free Entertainment, azienda Leader nel settore da oltre 15 anni, seleziona [JOB], con o senza esperienza per L'italia, Turchia, Tunisia ed Egitto.
Come [JOB] sarà in un contesto dinamico e fortemente meritocratico e dal team specializzato per facilitarne e favorirne l'inserimento.
[JOB] senza esperienza, l'agenzia offre la possibilità di imparare la professione, attraverso un percorso di formazione ed affiancamento retribuiti e a carico dell'azienda, con il quale imparare le skills e le conoscenze necessarie.
Per nuovo ufficio in Milano centro, che ha aperto le sue porte il primo Marzo 2024, siamo alla ricerca di [JOB] con ottime doti relazionali ed un network sviluppato nel cuore della città, per contribuire al nostro successo nel settore.
Ristorazione collettiva Jobtech, agenzia per il lavoro 100 digitale, è alla ricerca di [JOB] a Milano, in zona Linate, per conto di una nota azienda operante nel settore, per una sostituzione nelle giornate di* giovedì 22* e* venerdì 23*."""
old_texts = old_texts.split('\n')
# texts = [text.split(', ') for text in texts]
# texts = [[text,] for text in texts]