# Conteneur de serving FastAPI

Le conteneur exécute uniquement l'API FastAPI avec le modèle CatBoost V1 gelé.
Il ne contient aucune donnée DVF, base MLflow, archive d'artefacts, donnée
d'entraînement ou de test. Docker doit être installé avec un daemon accessible.

## Bundle autonome

Le modèle est exporté localement depuis le run MLflow validé :

```bash
python -m real_estate.ml.serving_bundle export \
  --run-id f140d75d05504aacad1ea18a09f0f4a4 \
  --output-dir artifacts/serving/v1
```

Cette commande ne lit aucune donnée DVF. Elle valide le run et le modèle gelés,
puis publie atomiquement :

```text
artifacts/serving/v1/
├── manifest.json
└── model.cbm
```

Le manifest décrit le contrat de features, les paramètres gelés, la provenance
du run et le SHA-256 du fichier `model.cbm`. Le bundle est local, ignoré par Git
et n'est pas incorporé à l'image.

## Construction et exécution

```bash
docker build -t real-estate-api:v1 .
```

```bash
docker run --rm \
  -p 8000:8000 \
  -e REAL_ESTATE_MODEL_BUNDLE_DIR=/app/model \
  --mount type=bind,source="$(pwd)/artifacts/serving/v1",target=/app/model,readonly \
  real-estate-api:v1
```

Le processus tourne avec l'utilisateur non-root `app`. Le bundle est monté en
lecture seule. Son absence, une modification de son manifest ou une divergence
du modèle fait échouer le démarrage. Le runtime ne consulte ni `mlflow.db` ni
`mlartifacts`, ne télécharge rien et ne réentraîne jamais le modèle.

Avec Compose :

```bash
docker compose up --build
```

Le healthcheck interne utilise uniquement la bibliothèque standard Python :

```bash
curl http://127.0.0.1:8000/health
```

## Prédiction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "surface_reelle_bati": 80.0,
    "nombre_pieces_principales": 4,
    "nombre_lots": 1,
    "surface_terrain": null,
    "has_dependance": true,
    "code_type_local": 1,
    "code_postal": "75001",
    "code_departement": "75",
    "source_code_commune": "75101",
    "canonical_commune_code": "75056",
    "resolved_geo_type": "ARM",
    "region_code": "11",
    "mutation_date": "2025-07-09"
  }'
```

Le modèle reste celui qui a été gelé et évalué selon le protocole V1. La
conteneurisation ne modifie ni ses features, ni ses hyperparamètres, ni ses
résultats officiels.
