# Qualification DVF V1

La couche `qualify.py` lit le Parquet normalisé et applique le contrat métier
existant de `clean.py`. Le grain d'entrée est **une ligne DVF normalisée** ; le
grain de sortie est **une mutation résidentielle simple admissible**. Toutes les
lignes d'une mutation sont présentées ensemble à `qualify_mutation`, puis
`build_observation` construit l'observation si elle est admissible.

`id_mutation` conserve le sens défini lors de la normalisation : il identifie un
bloc contigu synthétique, local à une publication. Il n'est ni un identifiant
juridique certifié, ni un identifiant stable entre publications DVF.

## Contrat métier réutilisé

Une mutation est admissible lorsque :

1. Toutes ses lignes ont `nature_mutation == "Vente"`.
2. `numero_disposition` est renseigné sur toutes les lignes et possède exactement
   une valeur distincte non vide.
3. `id_parcelle` est renseigné sur toutes les lignes et possède exactement une
   valeur distincte non vide.
4. Tous les `code_type_local` sont valides dans la taxonomie existante : 1, 2, 3
   ou 4. Une valeur absente ou hors taxonomie entraîne le rejet ; les libellés
   ne remplacent pas les codes.
5. Il existe exactement une ligne résidentielle de code 1 (Maison) ou 2
   (Appartement). Des lignes ressemblantes ne sont jamais dédupliquées.
6. Aucune ligne ne porte le code 4, local industriel, commercial ou assimilé.
7. `valeur_fonciere` est renseignée, numérique, finie et identique sur toutes les
   lignes. Les comparaisons conservent le comportement exact de `clean.py`.
8. La surface bâtie de l'unique ligne résidentielle est numérique, finie et
   strictement positive.
9. Le montant, la surface et leur ratio sont représentables comme des floats
   finis ; la surface reste positive après conversion.

Les dépendances de code 3 sont autorisées lorsque toutes les autres conditions
sont respectées. Leur présence est indiquée par `has_dependance`.

`prix_m2 = valeur_fonciere / surface_reelle_bati` est calculé seulement après
admission. Le numérateur est le montant global de la vente, dépendances
comprises ; le dénominateur est la surface de l'unique ligne résidentielle. Ce
ratio n'isole donc pas le prix du logement de celui de ses dépendances.

Il n'y a pas de filtrage d'outliers économiques dans cette couche. Le montant
est conservé comme donnée de transaction ; aucune sélection de features ML
n'est réalisée.

## Adaptation et schéma de sortie

`clean.py` et `validate.py` restent inchangés. Le Parquet normalisé ne contient
pas les colonnes `longitude` et `latitude`, alors que l'interface de `clean.py`
les exige. L'adaptateur les fournit avec la valeur `None` et les conserve nulles
en sortie. Aucun géocodage n'est effectué.

Les champs produits par `build_observation` sont réutilisés. `source_year`,
`nom_commune`, `nombre_lots` et `surface_terrain` sont copiés depuis l'unique ligne
résidentielle admissible. Les lots et la surface de terrain ne sont pas sommés
avec ceux des autres lignes de la mutation.

La sortie contient les 18 colonnes minimales demandées, complétées par
`source_row_count`, `longitude` et `latitude` provenant de l'interface existante :

| Colonnes | Type Arrow / Parquet |
| --- | --- |
| `source_year` | `int32` |
| `id_mutation`, `date_mutation`, `numero_disposition`, `id_parcelle` | `string` |
| `code_postal`, `nom_commune`, `code_departement`, `code_commune`, `type_local` | `string` |
| `code_type_local`, `nombre_pieces_principales`, `nombre_lots`, `source_row_count` | `int64` |
| `valeur_fonciere`, `surface_reelle_bati`, `prix_m2` | `float64`, finis |
| `surface_terrain` | `float64`, nullable |
| `longitude`, `latitude` | `float64`, nulls |
| `has_dependance` | `bool` |

Les métadonnées auxiliaires peuvent être nulles. `date_mutation` conserve la
date ISO du normalisé. `source_row_count` compte toutes les lignes de la mutation,
y compris les dépendances. Les champs d'adresse détaillés ne sont pas propagés.

## Intégrité et lecture progressive

Le fichier normalisé doit exister. Les noms de colonnes doivent être uniques et
les colonnes nécessaires présentes. Chaque ligne doit avoir l'année demandée,
un `source_row_number` entier positif strictement croissant et un `id_mutation`
non vide. Ces contrôles concernent aussi les mutations qui seront rejetées par
le contrat métier.

Les batches Parquet sont lus dans leur ordre. Le module accumule uniquement les
lignes de la mutation courante et qualifie le groupe complet quand l'identifiant
change. Cet état est conservé entre batches ; la dernière mutation est également
qualifiée après la fin de lecture. Un batch de sortie est écrit progressivement.
La mémoire consacrée aux lignes dépend des tailles des batches et de la plus
grande mutation, pas du nombre total de lignes de l'année.

Le contrôle des identifiants repose sur l'invariant produit par `normalize.py` :
le premier groupe porte `YYYY-1` et chaque nouveau groupe porte exactement le
numéro précédent augmenté de un, pour l'année demandée. Le même identifiant peut
occuper plusieurs lignes et traverser plusieurs batches. Les espaces extérieurs
sont retirés comme dans l'adaptateur initial. Les sauts, retours, réapparitions,
années incorrectes et formats non conformes provoquent une erreur d'intégrité.
Ce contrôle utilise uniquement l'identifiant courant et un compteur : sa mémoire
est constante, sans index SQLite ni collection des identifiants déjà rencontrés.
Il exige un fichier normalisé complet ; un extrait commençant à `YYYY-2` est refusé.

Chaque batch Arrow est converti par colonnes, puis parcouru sous forme de tuples,
sans dictionnaire par ligne. Les coordonnées `None` sont fournies directement au
constructeur de la DataFrame de chaque mutation. `clean.qualify_mutation` et
`clean.build_observation` restent appelées sans modification ; la seconde conserve
sa propre vérification d'admission. Les observations restent mises en tampon
avant écriture, dans l'ordre des mutations admissibles. Ces changements réduisent
des allocations et suppriment l'index disque ; le gain de temps sur l'année réelle
reste à mesurer.

Une erreur d'intégrité interrompt l'exécution ; elle n'est pas comptée comme un
rejet métier. Les validations de lignes sont progressives et doivent toutes
réussir avant la publication du fichier final.

## Comparaison logique de deux sorties

`real_estate.data.compare.compare_qualified_parquets` compare deux fichiers
existants par batches de taille bornée, sans les modifier. Elle retourne un
booléen et vérifie le schéma (noms, ordre, types et nullabilité), le nombre de
lignes et toutes les valeurs dans leur ordre. Elle accepte des découpages en
groupes de lignes différents ; compression et métadonnées descriptives ne font
pas partie de l'égalité logique.

Les nulls, booléens, entiers, chaînes et floats sont comparés sans tolérance
numérique. Deux NaN à la même position sont considérés égaux, mais distincts d'un
null ; les zéros signés sont égaux. Les erreurs de lecture sont propagées. Cette
comparaison ne se fonde pas sur le hash binaire des Parquet et ne relance aucune
qualification. Les compteurs d'exécution doivent être comparés séparément : ils
ne sont pas contenus dans les lignes du Parquet qualifié.

## Rejets et rapport d'exécution

Seul le premier motif renvoyé par `clean.py` est compté pour une mutation rejetée.
La priorité reste celle du contrat existant, dans l'ordre suivant :

| Motif de `ExclusionReason` | Valeur dans le rapport |
| --- | --- |
| `NOT_A_SALE` | `not_a_sale` |
| `INVALID_DISPOSITION` | `invalid_disposition` |
| `INVALID_PARCEL` | `invalid_parcel` |
| `INVALID_LOCAL_CODE` | `invalid_local_code` |
| `RESIDENTIAL_ROW_COUNT` | `residential_row_count` |
| `COMMERCIAL_OR_INDUSTRIAL_LOCAL` | `commercial_or_industrial_local` |
| `INVALID_VALUE` | `invalid_value` |
| `INCONSISTENT_VALUE` | `inconsistent_value` |
| `INVALID_RESIDENTIAL_SURFACE` | `invalid_residential_surface` |
| `UNREPRESENTABLE_FLOAT` | `unrepresentable_float` |

L'API retourne un rapport ; la CLI l'affiche en JSON. Il contient
`mutations_seen`, `mutations_admissible`, `mutations_rejected`, `retention_rate`
et `rejection_counts`. Le taux est le nombre de mutations admissibles divisé
par le nombre de mutations vues, entre 0 et 1 ; il vaut 0 pour une entrée vide.
Tous les motifs sont présents dans les compteurs, même lorsque leur compte
est nul. Aucun fichier séparé de rejets n'est écrit.

## Configuration, écriture et exécution

Les années autorisées sont celles de `dvf.years` dans `configs/data.yaml`.
L'entrée réutilise `dvf.normalization.output_directory`. La destination est
définie par `dvf.qualification.output_directory`, et la taille des batches par
`dvf.qualification.batch_size`. Les chemins restent relatifs au projet :

```text
data/interim/dvf/normalized/dvf_YYYY.parquet
  -> data/processed/dvf/qualified/dvf_YYYY.parquet
```

Le Parquet est écrit progressivement dans un fichier `.part`, situé dans le
répertoire de destination. Après lecture complète, son schéma et son nombre
de lignes sont vérifiés avant le remplacement atomique avec `os.replace`.
Un échec nettoie le temporaire et préserve l'éventuel fichier final préexistant.
Un résultat sans mutation admissible reste un Parquet valide avec le schéma
de sortie et zéro ligne. Les données traitées restent hors Git.

La commande suivante déclenche explicitement la qualification de l'année
demandée à partir du Parquet local :

```bash
python -m real_estate.data.qualify --year 2025
```

Une destination existante provoque une erreur par défaut. Son remplacement
nécessite l'option explicite :

```bash
python -m real_estate.data.qualify --year 2025 --force
```

L'import du module ne déclenche aucun traitement et cette couche n'effectue
aucun accès réseau. Les tests utilisent uniquement de petits Parquet synthétiques.
Le parcours est `raw -> normalized -> qualification V1 -> enrichment -> ML` ;
BPE, enrichissement et ML restent des étapes ultérieures.
