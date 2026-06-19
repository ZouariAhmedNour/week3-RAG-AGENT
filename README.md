# Week 3 — RAG Agent : Assistant qui lit des documents

Ce projet correspond à la **semaine 3** de ma roadmap Agentic AI.

Après avoir réalisé :

* **Week 1** : assistant de résumé avec sortie JSON structurée ;
* **Week 2** : agent avec tools / function calling ;

l’objectif de cette troisième semaine est de créer un agent capable de **répondre à partir de documents privés**.

Ce type d’agent utilise le principe du **RAG** :
**Retrieval-Augmented Generation**, c’est-à-dire génération augmentée par récupération d’informations.

L’agent ne doit pas répondre uniquement avec ses connaissances générales.
Il doit d’abord chercher les passages pertinents dans les documents fournis, puis utiliser ces passages pour générer une réponse avec des sources.

---

## Objectif du projet

Créer un assistant documentaire capable de répondre à des questions à partir de fichiers `.txt` ou `.pdf`.

Exemple de question :

```text
Explique-moi le rôle de Prisma dans HotelOps.
```

L’agent doit :

1. charger les documents depuis `data/docs/` ;
2. découper les documents en morceaux appelés chunks ;
3. générer des embeddings pour chaque chunk ;
4. stocker les chunks dans une base vectorielle ChromaDB ;
5. transformer la question utilisateur en embedding ;
6. rechercher les chunks les plus proches sémantiquement ;
7. envoyer ces chunks à Gemini comme contexte ;
8. répondre uniquement à partir des documents ;
9. afficher les sources utilisées.

---

## Ce que j’ai appris

Dans ce projet, j’ai appris à :

* comprendre le fonctionnement du RAG ;
* charger des fichiers `.txt` et `.pdf` ;
* découper un document en chunks ;
* utiliser un overlap pour ne pas couper les idées importantes ;
* générer des embeddings avec Gemini ;
* stocker des embeddings dans ChromaDB ;
* faire une recherche sémantique dans une base vectorielle ;
* construire un prompt avec contexte récupéré ;
* forcer l’agent à répondre uniquement à partir des documents ;
* afficher les sources utilisées dans la réponse ;
* tester le découpage des chunks avec `pytest` sans consommer l’API Gemini.

---

## Technologies utilisées

* Python
* Gemini API
* Google Gen AI SDK
* ChromaDB
* pypdf
* python-dotenv
* pytest
* PowerShell
* VS Code

---

## Notions importantes

### 1. RAG

RAG signifie :

```text
Retrieval-Augmented Generation
```

En français :

```text
Génération augmentée par récupération d’informations
```

Un chatbot classique fonctionne comme ceci :

```text
Question utilisateur
        ↓
LLM
        ↓
Réponse
```

Un système RAG fonctionne comme ceci :

```text
Question utilisateur
        ↓
Recherche dans les documents
        ↓
Passages pertinents
        ↓
LLM avec contexte
        ↓
Réponse avec sources
```

L’objectif est de réduire les hallucinations et d’obliger le modèle à s’appuyer sur des documents fournis.

---

### 2. Document

Un document est un fichier source utilisé par l’agent.

Exemples :

```text
cours-agentic-ai.txt
hotelops-backend.pdf
prisma-notes.txt
README.md
rapport.pdf
```

Dans ce projet, les documents sont placés dans :

```text
data/docs/
```

---

### 3. Chunk

Un chunk est un petit morceau de texte extrait d’un document.

Pourquoi découper les documents ?

Parce qu’un document complet peut être trop long à envoyer directement au modèle.
On le découpe donc en petits morceaux.

Exemple :

```text
Document complet
        ↓
Chunk 1
Chunk 2
Chunk 3
Chunk 4
```

Chaque chunk est ensuite transformé en embedding et stocké dans la base vectorielle.

---

### 4. Overlap

L’overlap est un chevauchement entre deux chunks.

Exemple :

```text
Chunk 1 : Prisma est utilisé pour gérer les modèles de données...
Chunk 2 : gérer les modèles de données et les relations avec PostgreSQL...
```

L’overlap évite de couper une idée importante entre deux morceaux.

Dans ce projet :

```python
chunk_size = 900
overlap = 150
```

---

### 5. Embedding

Un embedding est une représentation numérique d’un texte.

Exemple simplifié :

```text
"Prisma est un ORM pour PostgreSQL"
        ↓
[0.12, -0.45, 0.88, ...]
```

Les embeddings permettent de comparer le sens des textes.

Deux textes proches sémantiquement auront des embeddings proches :

```text
"Prisma communique avec PostgreSQL"
"Prisma est un ORM pour la base de données"
```

Même si les mots sont différents, le sens est proche.

---

### 6. Base vectorielle

Une base vectorielle stocke :

* le texte du chunk ;
* l’embedding du chunk ;
* la source du document ;
* l’index du chunk.

Dans ce projet, la base vectorielle utilisée est :

```text
ChromaDB
```

La base locale est stockée dans :

```text
chroma_db/
```

---

### 7. Recherche sémantique

Quand l’utilisateur pose une question, le système :

1. transforme la question en embedding ;
2. compare cet embedding avec les embeddings des chunks ;
3. récupère les chunks les plus proches ;
4. donne ces chunks au modèle Gemini.

Exemple :

```text
Question : "Explique-moi Prisma dans HotelOps"

Chunks récupérés :
- prisma-notes.txt / chunk 0
- hotelops-notes.txt / chunk 0
```

---

## Structure du projet

```text
week3-rag-agent/
│
├── .venv/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── pytest.ini
│
├── data/
│   └── docs/
│       ├── hotelops-notes.txt
│       └── prisma-notes.txt
│
├── chroma_db/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── loaders.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── indexer.py
│   └── rag_agent.py
│
└── tests/
    ├── __init__.py
    └── test_chunking.py
```

---

## Rôle des fichiers principaux

### `src/config.py`

Ce fichier centralise la configuration du projet.

Il permet de :

* charger les variables d’environnement depuis `.env` ;
* récupérer les valeurs comme `GEMINI_API_KEY`, `GEMINI_MODEL`, `CHROMA_PATH` ;
* créer le client Gemini.

---

### `src/loaders.py`

Ce fichier sert à charger les documents.

Il contient les fonctions pour lire :

* les fichiers `.txt` ;
* les fichiers `.pdf`.

Les documents sont chargés depuis :

```text
data/docs/
```

---

### `src/chunking.py`

Ce fichier contient la logique de découpage des documents.

Fonction principale :

```python
chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]
```

Elle transforme un long texte en plusieurs chunks plus petits.

---

### `src/embeddings.py`

Ce fichier sert à générer les embeddings avec Gemini.

Fonction principale :

```python
embed_text(text: str) -> list[float]
```

Elle prend un texte et retourne un vecteur numérique.

---

### `src/indexer.py`

Ce fichier sert à indexer les documents dans ChromaDB.

Il fait les étapes suivantes :

1. charger les documents ;
2. découper les documents en chunks ;
3. générer un embedding pour chaque chunk ;
4. stocker les chunks et les embeddings dans ChromaDB.

Commande utilisée :

```bash
python -m src.indexer --reset
```

---

### `src/rag_agent.py`

Ce fichier contient l’agent RAG.

Il permet de :

1. recevoir une question utilisateur ;
2. générer l’embedding de la question ;
3. chercher les chunks les plus pertinents dans ChromaDB ;
4. construire un prompt avec le contexte récupéré ;
5. appeler Gemini ;
6. afficher la réponse avec les sources.

Commande utilisée :

```bash
python -m src.rag_agent --ask "Explique-moi le rôle de Prisma dans HotelOps."
```

---

### `tests/test_chunking.py`

Ce fichier contient les tests unitaires du découpage en chunks.

Les tests vérifient :

* qu’un texte long est bien découpé en plusieurs chunks ;
* qu’un petit texte reste en un seul chunk ;
* que `chunk_size` doit être positif ;
* que `overlap` doit être inférieur à `chunk_size`.

---

## Installation

### 1. Créer l’environnement virtuel

```bash
python -m venv .venv
```

### 2. Activer l’environnement virtuel

Sur Windows PowerShell :

```bash
.\.venv\Scripts\Activate.ps1
```

Résultat attendu :

```text
(.venv) PS D:\AgenticAI\week3-rag-agent>
```

### 3. Installer les dépendances

```bash
python -m pip install --upgrade pip
pip install google-genai python-dotenv chromadb pypdf pytest
```

### 4. Sauvegarder les dépendances

```bash
pip freeze > requirements.txt
```

---

## Configuration `.env`

Créer un fichier `.env` à la racine du projet :

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=gemini-embedding-001
CHROMA_PATH=chroma_db
COLLECTION_NAME=course_docs
```

Le fichier `.env` ne doit jamais être envoyé sur GitHub.

---

## Exemple de `.gitignore`

```gitignore
.venv/
.env
__pycache__/
.pytest_cache/
chroma_db/
```

Explication :

* `.venv/` : environnement virtuel local ;
* `.env` : contient la clé API ;
* `__pycache__/` : cache Python ;
* `.pytest_cache/` : cache des tests ;
* `chroma_db/` : base vectorielle générée localement.

---

## Ajouter des documents

Les documents doivent être placés dans :

```text
data/docs/
```

Exemple :

```text
data/docs/hotelops-notes.txt
data/docs/prisma-notes.txt
```

Exemple de contenu pour `hotelops-notes.txt` :

```text
HotelOps est une application de gestion de maintenance hôtelière.
Le backend utilise Express, Prisma et PostgreSQL.
Prisma sert d’ORM pour définir les modèles de données comme User, Role, Location et MaintenanceTicket.
Le modèle MaintenanceTicket contient les informations liées aux tickets de maintenance : titre, description, priorité, statut, localisation et utilisateur assigné.
L’objectif est de permettre à la réception, au chef maintenance et aux agents de suivre les incidents en temps réel.
```

Exemple de contenu pour `prisma-notes.txt` :

```text
Prisma est utilisé pour communiquer avec PostgreSQL.
Le fichier schema.prisma décrit les tables, les relations et les contraintes de la base de données.
Après modification du schéma, on utilise les migrations Prisma pour appliquer les changements à la base.
Dans HotelOps, Prisma permet de relier les tickets aux utilisateurs, aux localisations, aux catégories et aux statuts.
```

---

## Indexer les documents

Avant de poser des questions, il faut indexer les documents :

```bash
python -m src.indexer --reset
```

Résultat attendu :

```text
[INDEXED] data/docs/hotelops-notes.txt - chunk 0
[INDEXED] data/docs/prisma-notes.txt - chunk 0

Indexation terminée : 2 chunks ajoutés.
```

Cette commande crée ou met à jour la base vectorielle locale :

```text
chroma_db/
```

---

## Poser une question à l’agent RAG

Commande :

```bash
python -m src.rag_agent --ask "Explique-moi le rôle de Prisma dans HotelOps."
```

Exemple de réponse attendue :

```text
--- Réponse RAG ---

Prisma est utilisé dans HotelOps comme ORM pour communiquer avec PostgreSQL.
Il permet de définir les modèles de données dans le fichier schema.prisma et de gérer les relations entre les tickets, les utilisateurs, les localisations, les catégories et les statuts.

Sources récupérées :
- data/docs/prisma-notes.txt / chunk 0
- data/docs/hotelops-notes.txt / chunk 0
```

---

## Tester une question hors documents

Commande :

```bash
python -m src.rag_agent --ask "Quel est le meilleur framework frontend en 2026 ?"
```

Réponse attendue :

```text
Je ne trouve pas cette information dans les documents fournis.
```

C’est un comportement important.
Un bon agent RAG ne doit pas inventer une réponse quand les documents ne contiennent pas l’information.

---

## Lancer les tests

Commande :

```bash
python -m pytest
```

Résultat attendu :

```text
4 passed
```

Les tests ne consomment pas l’API Gemini.
Ils testent seulement la logique locale de découpage des textes.

---

## Configuration `pytest.ini`

Créer un fichier `pytest.ini` à la racine :

```ini
[pytest]
pythonpath = .
testpaths = tests
```

Cela permet à `pytest` de reconnaître correctement le dossier `src`.

---

## Flow complet du RAG

Le fonctionnement complet du projet est :

```text
Documents dans data/docs/
        ↓
Chargement des fichiers
        ↓
Découpage en chunks
        ↓
Génération des embeddings
        ↓
Stockage dans ChromaDB
        ↓
Question utilisateur
        ↓
Embedding de la question
        ↓
Recherche sémantique
        ↓
Récupération des meilleurs chunks
        ↓
Prompt Gemini avec contexte
        ↓
Réponse finale avec sources
```

---

## Exemple visuel du processus

```text
Question :
"Explique-moi Prisma dans HotelOps"

        ↓

Embedding de la question

        ↓

Recherche dans ChromaDB

        ↓

Chunks récupérés :
- prisma-notes.txt / chunk 0
- hotelops-notes.txt / chunk 0

        ↓

Gemini reçoit :
- la question
- les chunks pertinents
- les règles de réponse

        ↓

Réponse :
"Prisma est utilisé comme ORM..."

        ↓

Sources :
- prisma-notes.txt
- hotelops-notes.txt
```

---

## Bonnes pratiques RAG

### 1. Toujours afficher les sources

Un agent RAG doit afficher les documents utilisés.

Bon exemple :

```text
Sources :
- data/docs/prisma-notes.txt / chunk 0
```

Mauvais exemple :

```text
Réponse sans source.
```

Les sources permettent de vérifier que l’agent répond bien à partir des documents.

---

### 2. Ne pas récupérer trop de chunks

Si on récupère trop de chunks, le modèle reçoit trop d’informations inutiles.

Bon début :

```text
top_k = 3
```

À tester ensuite :

```text
top_k = 5
```

À éviter au début :

```text
top_k = 20
```

Trop de contexte peut rendre la réponse moins précise.

---

### 3. Bien choisir la taille des chunks

Pour des notes simples :

```text
chunk_size = 700 à 1000
overlap = 100 à 200
```

Pour du code :

```text
chunk_size = 1200 à 1800
overlap = 200
```

Pour des définitions courtes :

```text
chunk_size = 400 à 700
overlap = 100
```

---

### 4. Commencer avec des fichiers `.txt`

Les fichiers `.txt` sont simples à lire.

Les PDF peuvent poser des problèmes :

* PDF scanné ;
* texte en colonnes ;
* tableaux ;
* images ;
* mauvais ordre d’extraction.

Pour débuter :

```text
TXT d'abord
PDF simples ensuite
PDF complexes plus tard
```

---

### 5. Tester avec des questions pièges

Il faut tester le RAG avec plusieurs types de questions :

```text
Question dont la réponse existe dans les documents.
Question dont la réponse n’existe pas dans les documents.
Question vague.
Question avec des mots différents mais un sens proche.
Question qui mélange deux sujets.
```

Exemples :

```text
Explique le rôle de Prisma dans HotelOps.
```

```text
Quel modèle React Native est utilisé dans le projet ?
```

```text
Quel est le meilleur framework frontend en 2026 ?
```

---

### 6. Ne pas pousser les documents sensibles

Si les documents contiennent des informations privées, il ne faut pas les envoyer sur GitHub.

Exemple :

```text
logs privés
clés API
données clients
documents internes sensibles
```

Dans ce cas, garder seulement des documents d’exemple dans `data/docs/`.

---

## Erreurs fréquentes

### Erreur : `No module named chromadb`

Solution :

```bash
pip install chromadb
```

---

### Erreur : `No module named pypdf`

Solution :

```bash
pip install pypdf
```

---

### Erreur : `GEMINI_API_KEY` manquante

Vérifier que `.env` existe à la racine :

```text
week3-rag-agent/.env
```

Et contient :

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

### Erreur : collection inexistante

Cela arrive si l’agent RAG est lancé avant l’indexation.

Solution :

```bash
python -m src.indexer --reset
```

Puis relancer :

```bash
python -m src.rag_agent --ask "Explique Prisma"
```

---

### Réponse vague ou incorrecte

Causes possibles :

* documents insuffisants ;
* mauvais découpage en chunks ;
* `top_k` trop faible ;
* PDF mal extrait ;
* question trop vague.

Solutions possibles :

```text
augmenter top_k à 5
ajouter plus de documents
améliorer les fichiers .txt
ajuster chunk_size
afficher les chunks récupérés pour debug
```

---

## Commandes utiles

Créer l’environnement virtuel :

```bash
python -m venv .venv
```

Activer l’environnement :

```bash
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances :

```bash
pip install google-genai python-dotenv chromadb pypdf pytest
```

Sauvegarder les dépendances :

```bash
pip freeze > requirements.txt
```

Indexer les documents :

```bash
python -m src.indexer --reset
```

Poser une question :

```bash
python -m src.rag_agent --ask "Explique-moi le rôle de Prisma dans HotelOps."
```

Changer le nombre de chunks récupérés :

```bash
python -m src.rag_agent --ask "Explique Prisma" --top-k 5
```

Lancer les tests :

```bash
python -m pytest
```

---

## Checklist Week 3

À la fin de cette semaine, le projet doit valider :

```text
[ ] Projet week3-rag-agent créé
[ ] Environnement virtuel .venv activé
[ ] Gemini configuré dans .env
[ ] ChromaDB installé
[ ] Documents ajoutés dans data/docs/
[ ] Chargement TXT fonctionnel
[ ] Chargement PDF fonctionnel
[ ] Découpage en chunks fonctionnel
[ ] Embeddings Gemini fonctionnels
[ ] Indexation dans ChromaDB fonctionnelle
[ ] Recherche sémantique fonctionnelle
[ ] Réponse Gemini avec contexte
[ ] Sources affichées
[ ] Tests pytest fonctionnels
[ ] Tests sans consommation API
```

---

## Résultat final de la Week 3

À la fin de ce projet, j’ai construit un assistant documentaire intelligent.

Il est capable de :

* lire des documents ;
* indexer leur contenu ;
* comprendre une question ;
* rechercher les passages pertinents ;
* générer une réponse à partir du contexte ;
* afficher les sources utilisées.

Ce projet représente une étape importante vers la création d’agents IA utiles dans des cas réels.

---

## Compétence acquise

Cette Week 3 m’a permis de comprendre la base du RAG :

```text
Documents + Chunks + Embeddings + Vector DB + LLM = Assistant documentaire intelligent
```

---

## Lien avec les semaines précédentes

### Week 1

```text
Texte → Gemini → JSON structuré
```

J’ai appris à obtenir une réponse structurée et exploitable.

### Week 2

```text
Texte → Gemini → Tools Python → Réponse finale
```

J’ai appris à créer un agent capable d’appeler des fonctions.

### Week 3

```text
Question → Recherche documents → Contexte → Gemini → Réponse avec sources
```

J’ai appris à créer un agent capable de répondre à partir de documents privés.

---

## Prochaine étape

La prochaine étape de la roadmap est :

```text
Week 4 — Agent avec mémoire
```

Objectif :

Créer un agent capable de retenir des informations importantes comme :

* préférences utilisateur ;
* historique de conversation ;
* tâches déjà faites ;
* contexte d’un projet ;
* difficultés rencontrées.

Ce sera une étape importante pour passer d’un agent documentaire à un agent plus personnalisé et durable.

---

## Application future avec HotelOps

Ce projet prépare directement le futur agent IA pour HotelOps.

Exemples d’utilisation future :

```text
Explique-moi pourquoi la création d’un ticket échoue avec Prisma.
```

```text
Résume-moi la logique du service ticketService.ts.
```

```text
Cherche dans la documentation backend comment fonctionne l’assignation des agents.
```

```text
Explique les relations entre User, MaintenanceTicket, Location et Status.
```

L’agent pourra lire :

* `schema.prisma` ;
* les services backend ;
* les routes Express ;
* les logs ;
* les fichiers README ;
* la documentation API.

Puis il répondra avec des sources au lieu d’inventer.
