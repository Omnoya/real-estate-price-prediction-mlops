# Intégration continue

Le workflow GitHub Actions `.github/workflows/ci.yml` s'exécute lors des pushes
sur `main` et pour chaque pull request. Ses permissions sont limitées à la
lecture du contenu du dépôt.

## Qualité Python

Le job `quality` utilise Python 3.12 et le cache pip fourni par
`actions/setup-python`. Il installe le projet avec l'extra `dev`, puis exécute :

```text
python -m pip check
python -m ruff check .
python -m pytest -q
```

Les tests reposent uniquement sur de petites fixtures synthétiques et sur des
répertoires temporaires. Ils ne nécessitent aucune donnée DVF, base MLflow,
archive d'artefacts ou bundle réel.

## Image de serving

Le job `docker` ne commence qu'après le succès de `quality`. Il construit
l'image locale `real-estate-api:ci`, vérifie que son utilisateur configuré est
exactement `app`, confirme que MLflow n'est pas installé dans l'image, puis
importe l'application FastAPI sans lancer son lifespan.

L'image n'est jamais poussée vers un registre. Elle n'est pas démarrée avec
`/health` ou `/predict`, car ces routes nécessitent le bundle autonome monté au
runtime et aucun bundle réel n'est disponible en CI.
