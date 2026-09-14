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

## Premier modèle CatBoost

Le premier modèle ML est un `CatBoostRegressor`. Ce choix permet de transmettre
directement les sept variables catégorielles du contrat à CatBoost, sans créer
de one-hot encoding global ni apprendre un encodage manuel. Les catégories
nulles prennent la valeur stable `__MISSING__`. Les valeurs numériques absentes
restent des `NaN`, que CatBoost traite nativement, et les booléens sont convertis
de façon déterministe en 0/1.

Le premier canari utilise les paramètres fixes suivants : 1 000 itérations,
un taux d'apprentissage de 0,05, une profondeur de 8 et la graine 42. La
validation temporelle 2024 est fournie comme `eval_set` avec un early stopping
de 100 itérations et `use_best_model=True`. Elle ne sert à calculer aucune
feature, catégorie ou statistique de préparation. Aucun tuning n'est réalisé à
ce stade et CatBoost est configuré pour ne pas écrire ses fichiers de suivi
locaux.

Le rapport compare les trois métriques globales au benchmark validé
`DEPARTMENT_TYPE_MEDIAN` : RMSE log 0,6590399945589221, MAE 1 147,7414970737116
€/m² et erreur absolue médiane 754,010441767065 €/m². `absolute_delta` vaut
`model_value - baseline_value` ; une valeur négative indique donc une
amélioration. Le pourcentage d'amélioration est positif lorsque le modèle fait
mieux.

La commande CatBoost partage le même chargeur de développement que les
baselines : elle ouvre seulement 2021--2023 pour le fit et 2024 pour
l'évaluation. Elle n'expose aucune option permettant de lire ou d'évaluer 2025.
Le test final reste scellé jusqu'au gel séparé du modèle.
