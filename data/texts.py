import pandas as pd

TEXT_JOBS_v0 = [
('Studio Legale Rossi & Associati cerca [JOB] Legale',"TODO"),
('Lo Studio Legale Rossi & Associati, rinomato studio con sede nel centro di Milano specializzato in diritto societario e commerciale, è alla ricerca di una [JOB] Legale per unirsi al nostro team dinamico e in crescita.',"TODO"),
('Descrizione del ruolo:',"INCLUSIVO"),
("[JOB] svolgerà un ruolo chiave nel supportare le attività quotidiane dello studio, garantendo un'efficiente gestione amministrativa e contribuendo al successo complessivo del team legale.","TODO"),
('Responsabilità principali:',"INCLUSIVO"),
("Gestione dell'agenda e organizzazione degli appuntamenti per gli avvocati dello studio","NON INCLUSIVO"),
('Accoglienza dei clienti e gestione delle comunicazioni telefoniche ed email',"NON INCLUSIVO"),
('Preparazione e formattazione di documenti legali, lettere e relazioni',"INCLUSIVO"),
('Archiviazione e gestione della documentazione fisica e digitale',"INCLUSIVO"),
('Supporto nella preparazione di fascicoli per udienze e riunioni',"INCLUSIVO"),
('Coordinamento con altri professionisti e studi legali',"NON INCLUSIVO"),
('Gestione delle pratiche amministrative e contabili di base',"INCLUSIVO"),
('Organizzazione di viaggi e trasferte per i professionisti dello studio',"NON INCLUSIVO"),
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
# 
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
("Sarai contattato per un colloquio conoscitivo. ","NON INCLUSIVO"),
("Sarai contattatoa per un colloquio conoscitivo. ","NON INCLUSIVO"),
("Si prega di inviare il proprio curriculum vitae indicando nell'oggetto del messaggio il titolo della posizione per cui si candida.","INCLUSIVO"),
("Grazie per l'interesse dimostrato. ","INCLUSIVO"),
("I candidati, nel rispetto del D.lgs. 198/2006, D.lgs 215/2003 e D.lgs 216/2003, sono invitati a leggere l'informativa sulla privacy (Regolamento UE n. 2016/679).","NON INCLUSIVO"),
# 
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

JOBS_V0 = pd.read_csv('data/synt/jobs.csv').values.tolist()

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