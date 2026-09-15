# API de prédiction

L'API FastAPI sert le modèle CatBoost V1 gelé. Elle estime un prix au mètre
carré en reconvertissant la prédiction `log_prix_m2` avec `exp`. Elle ne prédit
pas la valeur foncière totale et ne lit aucun fichier DVF au runtime.

## Configuration locale

Le modèle est chargé une seule fois au démarrage depuis le run MLflow validé.
Les valeurs par défaut sont :

- `REAL_ESTATE_MLFLOW_TRACKING_URI=sqlite:///mlflow.db`
- `REAL_ESTATE_MODEL_RUN_ID=f140d75d05504aacad1ea18a09f0f4a4`

Quand `REAL_ESTATE_MODEL_BUNDLE_DIR` est défini, le bundle autonome est
prioritaire et MLflow n'est pas consulté. Un bundle absent ou invalide fait
échouer le démarrage de l'application.

Ces variables permettent de sélectionner un autre emplacement local et un run
compatible. Avant de servir, l'application contrôle la version `v1`, le statut
gelé du protocole, la target, le type CatBoost, le hash du contrat de features,
les paramètres du modèle et l'ordre exact des 16 features.

## Démarrage

```bash
python -m uvicorn real_estate.api.main:app --host 127.0.0.1 --port 8000
```

La documentation Swagger/OpenAPI est disponible sur `/docs`.

```bash
curl http://127.0.0.1:8000/health
```

## Endpoints

- `GET /health` indique si le modèle est prêt et retourne sa version.
- `GET /model` expose le type de modèle, sa version, sa target et son nombre de
  features. Aucun chemin ou URI d'artefact n'est exposé.
- `POST /predict` valide les données métier, dérive les features temporelles et
  les indicateurs de valeurs manquantes, puis retourne
  `predicted_price_m2` et `model_version`.

Exemple :

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

```json
{
  "predicted_price_m2": 8421.37,
  "model_version": "v1"
}
```

Une `surface_terrain` nulle signifie que la mesure manque : CatBoost reçoit
`NaN` et l'indicateur `surface_terrain_missing` vaut vrai. Une valeur égale à
zéro reste une valeur mesurée et l'indicateur vaut faux.

Une géographie non résolue est acceptée avec `resolved_geo_type` nul. Les
catégories géographiques nulles deviennent la catégorie stable `__MISSING__`,
comme lors de l'entraînement. L'API n'invente pas de géographie et n'effectue
aucun lookup réseau.
