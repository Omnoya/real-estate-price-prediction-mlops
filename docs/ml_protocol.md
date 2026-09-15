# Protocole ML V1

## Découpage temporel

Le développement utilise les transactions 2021, 2022 et 2023 pour
l'apprentissage et les transactions 2024 pour la validation. Aucun tirage
aléatoire n'est effectué. Le test final 2025 a été maintenu scellé pendant la
sélection : les commandes de baseline ne disposent d'aucune option qui le
charge ou l'évalue.

Les volumes de sélection sont 2 048 075 lignes de train et 521 169 lignes de
validation. Le test final contient 591 274 lignes.

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

## Test final fermé

La commande `train` ne possède aucun chemin vers 2025. La commande explicite
`evaluate --run-id` a ouvert une seule fois les 591 274 observations du test,
après vérification du run gelé et rechargement de son modèle depuis MLflow.

Les résultats définitifs 2025 sont :

- global : RMSE log 0,5142518638253868, MAE 841,7623633027805 €/m² et erreur
  absolue médiane 497,8948115618889 €/m² ;
- maisons : RMSE log 0,5162246411847635, MAE 657,7340904677158 €/m² et erreur
  absolue médiane 436,505573319492 €/m² ;
- appartements : RMSE log 0,5128061726001529, MAE 976,1746318557172 €/m² et
  erreur absolue médiane 556,1008610844178 €/m².

La validation 2024 a servi exclusivement à sélectionner et geler le modèle. Le
test 2025 mesure sa performance finale après réentraînement sur 2021--2024 ; il
ne constitue pas une nouvelle validation.

Les tags MLflow `test_evaluation_started=true` et `test_evaluated=true`
ferment définitivement ce test. Il n'existe pas d'option de contournement.
Aucune modification ni ré-optimisation du modèle V1 à partir des résultats
2025 n'est autorisée. Le résultat complet est conservé dans
`reports/final_test_2025.json`.
