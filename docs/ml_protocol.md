# Protocole ML V1

## Découpage temporel

Le développement utilise les transactions 2021, 2022 et 2023 pour
l'apprentissage et les transactions 2024 pour la validation. Aucun tirage
aléatoire n'est effectué. Le test final 2025 reste scellé : les commandes de
baseline ne disposent d'aucune option qui le charge ou l'évalue.

Les volumes attendus lors de l'exécution réelle sont 2 048 075 lignes de train,
521 169 lignes de validation et 591 274 lignes de test. Cette implémentation ne
lit et ne valide pas encore le test réel.

## Target

La source est `prix_m2`. L'apprentissage et la RMSE utilisent
`log_prix_m2 = ln(prix_m2)`. Chaque valeur doit être renseignée, finie et
strictement positive. Les prédictions sont reconverties avec `exp` avant les
métriques en euros par mètre carré.

La baseline V1 n'applique aucun clipping, winsorizing ou filtrage d'outlier.
Les règles éventuelles seront décidées séparément et apprises sur le train.

## Feature allowlist

Le dataset ML expose exactement les variables suivantes :

- numériques : `surface_reelle_bati`, `nombre_pieces_principales`,
  `nombre_lots`, `surface_terrain` ;
- booléenne : `has_dependance` ;
- catégorielles : `code_type_local`, `code_postal`, `code_departement`,
  `source_code_commune`, `canonical_commune_code`, `resolved_geo_type`,
  `region_code` ;
- dérivées : `mutation_year`, `mutation_month`,
  `surface_terrain_missing`, `geography_unresolved`.

Les catégories et les valeurs manquantes restent brutes pour ces baselines.
Le futur pipeline devra gérer explicitement les catégories inconnues. Les 1 686
géographies non résolues observées sur 2021--2025 restent dans le dataset et
sont signalées par `geography_unresolved`. `surface_terrain` manque sur environ
60 % des lignes ; sa nullité n'est jamais assimilée à une surface nulle.

Toutes les autres colonnes sont exclues par construction. En particulier,
`prix_m2` et `valeur_fonciere` créeraient une fuite directe. Les identifiants
`id_mutation`, `id_parcelle`, `numero_disposition`, les métadonnées de pipeline
`source_row_count`, `source_year`, `cog_year`, les dates/libellés bruts et les
coordonnées ne sont pas des features V1. `department_code` est également exclu
au profit du code département DVF déclaré dans l'allowlist.

## Baselines et métriques

`GLOBAL_MEDIAN` prédit partout la médiane de `log_prix_m2` calculée sur le train.
`DEPARTMENT_TYPE_MEDIAN` calcule sur le train une médiane par
`(code_departement, code_type_local)` et utilise la médiane globale train pour
un groupe inconnu. Aucune statistique de validation n'entre dans leur fit.

La validation 2024 rapporte globalement, puis séparément pour maisons et
appartements :

- la RMSE entre targets et prédictions logarithmiques ;
- la MAE en euros par mètre carré après `exp` ;
- la médiane des erreurs absolues en euros par mètre carré.

La MAPE n'est pas utilisée, car des targets valides mais très proches de zéro
la rendent trompeuse. Aucune métrique 2025 ne sera produite avant le gel du
modèle final.

## Modèle CatBoost V1 gelé

Le modèle retenu est un `CatBoostRegressor`. Il reçoit directement les sept
variables catégorielles du contrat, sans one-hot encoding global ni encodage
appris manuellement. Les catégories nulles prennent la valeur stable
`__MISSING__`, les valeurs numériques absentes restent des `NaN` et les booléens
sont convertis de façon déterministe en 0/1.

La sélection a utilisé 2021--2023 pour l'apprentissage et 2024 pour la
validation temporelle. Avec un budget de 3 000 itérations, la meilleure
itération observée était 2 999. Les métriques officielles de validation sont :

- RMSE log : 0,5454508466588915 ;
- MAE : 829,2568229919669 €/m² ;
- erreur absolue médiane : 494,46181031372157 €/m².

La baseline `DEPARTMENT_TYPE_MEDIAN` obtenait respectivement
0,6590399945589221, 1 147,7414970737116 €/m² et 754,010441767065 €/m². Le détail
versionné de la sélection se trouve dans `reports/model_selection.json`.

Le tuning est terminé. Les features, la target, le traitement des valeurs
extrêmes, l'architecture et les hyperparamètres ne peuvent plus être modifiés à
partir des résultats 2024. Le modèle V1 utilise `RMSE`, 3 000 itérations, un
taux d'apprentissage de 0,05, une profondeur de 8, la graine 42 et désactive les
fichiers de suivi CatBoost.

## Entraînement final et MLflow

Le fit final réunit 2021, 2022, 2023 et 2024, soit 2 569 244 observations
attendues. Il exécute exactement 3 000 itérations. Il n'utilise ni `eval_set` ni
early stopping : 2024 appartient désormais au train final et le budget a été
gelé lors de la sélection. Aucune statistique de 2025 n'intervient dans ce fit.

Chaque entraînement final crée un run dans un store SQLite MLflow local et place
ses artefacts dans un répertoire local séparé. Le run consigne le protocole, les
hyperparamètres, les versions logicielles, le commit Git, le contrat ordonné des
features, la configuration, le rapport de sélection et le modèle CatBoost. Les
données DVF ne sont jamais loggées. Le store, le modèle et les artefacts générés
restent hors Git.

## Test final scellé

La commande `train` ne possède aucun chemin vers 2025. Seule la commande
explicite `evaluate --run-id` peut ouvrir les 591 274 observations du test,
après avoir vérifié le run gelé et rechargé son modèle depuis MLflow. Elle
calcule les trois métriques globalement, puis pour maisons et appartements, et
les ajoute au run.

Le tag MLflow `test_evaluated=true` interdit une seconde évaluation du même run.
Il n'existe pas d'option de contournement. Après l'observation des métriques
2025, aucune modification ni ré-optimisation du modèle V1 n'est autorisée.
