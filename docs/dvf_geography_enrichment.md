# Enrichissement géographique annuel des DVF qualifiés

`real_estate.data.enrich_geography` ajoute le référentiel COG annuel aux mutations
DVF déjà qualifiées. Le grain reste une mutation résidentielle simple admissible.
Chaque ligne DVF est conservée exactement une fois, dans son ordre initial, avec
toutes ses colonnes inchangées. Cette étape n'ajoute ni BPE ni features ML.

## Entrées et configuration

Les chemins relatifs au projet sont configurés dans
`dvf.geography_enrichment` de `configs/data.yaml` :

| Paramètre | Template |
| --- | --- |
| `qualified_input` | `data/processed/dvf/qualified/dvf_{year}.parquet` |
| `geography_input` | `data/interim/cog/geography/cog_geography_{year}.parquet` |
| `output` | `data/processed/dvf/geography/dvf_{year}.parquet` |

Les années autorisées viennent de `dvf.years` et doivent également être présentes
dans `cog.years`. Chaque année DVF utilise exclusivement le COG du même millésime.
La taille de batch configurée est de 50 000 lignes. Les Parquet sources ne sont
jamais modifiés et la sortie doit être distincte des deux entrées. Les données
sous `data/processed` restent hors Git.

## Jointure et valeurs non résolues

La jointure gauche utilise uniquement :

```text
DVF.code_commune = geography.source_code_commune
```

Le référentiel a déjà appliqué la priorité COM → ARM → COMD → COMA. Le pipeline
d'enrichissement réutilise sa validation et ne réimplémente aucune résolution
hiérarchique. Les codes sont des chaînes dont les zéros initiaux sont conservés.
Les libellés et les départements ne servent jamais de clés de secours. Aucun
historique communal, fichier de mouvements ou autre source n'est consulté.

Lorsqu'un code existe dans le référentiel, ses neuf champs géographiques sont
copiés. Le code source doit être égal au code DVF, les années doivent coïncider,
et le département canonique doit être égal à `DVF.code_departement`. Toute
divergence d'année ou de département provoque une erreur explicite.

Lorsqu'un code est absent, la mutation reste présente et devient `unresolved`,
défini exactement par `resolved_geo_type is null`. `source_code_commune` conserve
`DVF.code_commune` et `cog_year` conserve le millésime consulté. Les sept autres
champs géographiques sont nulls. Aucune tentative de remapping n'est effectuée.

## Schéma de sortie

La sortie comporte exactement 30 colonnes : les 21 colonnes qualifiées dans leur
ordre original, suivies des neuf champs ci-dessous dans cet ordre. Les types et
les valeurs des colonnes DVF sont conservés.

| Champ ajouté | Type Arrow | Nullabilité et contenu |
| --- | --- | --- |
| `source_code_commune` | `string` | Toujours renseigné : code commune DVF inchangé |
| `resolved_geo_type` | `string` | COM, ARM, COMD ou COMA si résolu ; null sinon |
| `source_geo_label` | `string` | Libellé source du référentiel si résolu ; null sinon |
| `parent_commune_code` | `string` | Parent pour ARM/COMD/COMA ; null pour COM et unresolved |
| `canonical_commune_code` | `string` | Code canonique du référentiel si résolu ; null sinon |
| `canonical_commune_label` | `string` | Libellé canonique si résolu ; null sinon |
| `region_code` | `string` | Région de la COM canonique si résolu ; null sinon |
| `department_code` | `string` | Département de la COM canonique si résolu ; null sinon |
| `cog_year` | `int32` | Toujours renseigné : millésime COG consulté |

Seuls `source_code_commune` et `cog_year` sont non nullables parmi les neuf champs
ajoutés. La nullabilité de la sortie enrichie diffère donc de celle du référentiel
COG autonome, qui ne comporte aucune entrée unresolved.

## Validation et exécution en mémoire bornée

Le référentiel, environ 37 000 codes, est chargé une fois en mémoire. Ses neuf
colonnes, types, clés uniques, millésime et invariants canoniques sont vérifiés
avec la validation du module `geography`. Un référentiel incohérent est refusé
avant le parcours complet du DVF.

Les mutations DVF sont lues par batches PyArrow. Le pipeline conserve les colonnes
Arrow d'entrée et ajoute les neuf colonnes obtenues par lookup dans l'index du
référentiel. Il ne crée pas de DataFrame Pandas par mutation et ne charge pas
l'année DVF complète en mémoire.

Chaque batch vérifie les colonnes requises, le millésime, les codes commune
non nuls de longueur cinq et les identifiants de mutation renseignés. Les règles
d'admissibilité de la qualification V1 ne sont pas réévaluées. Les validations
de sortie vérifient aussi l'égalité des codes et des années, les départements
résolus et les nulls exacts des mutations unresolved.

La validation du Parquet temporaire relit la sortie et le DVF par batches : elle
compare les 21 colonnes sources, leur ordre et leurs valeurs, et contrôle le
schéma final ainsi que les invariants géographiques. Le nombre de lignes doit
être non nul et identique en entrée et en sortie. Cette comparaison vérifie
qu'aucune perte, duplication ou permutation de mutation n'a été introduite.

## Publication et CLI

```bash
python -m real_estate.data.enrich_geography --year 2025
python -m real_estate.data.enrich_geography --year 2025 --force
```

Ces commandes sont prévues pour une exécution explicite après revue. Elles ne
sont jamais déclenchées à l'import. Les tests utilisent uniquement de petits
Parquet synthétiques ; ils ne réalisent aucun enrichissement des données réelles.

L'écriture se fait dans `dvf_YYYY.parquet.part`, créé en mode exclusif dans le
répertoire de destination. Un `.part` préexistant est refusé. Le writer est fermé,
le Parquet temporaire est entièrement validé et son contenu synchronisé par
`fsync` avant publication atomique avec `os.replace`.

Un fichier final existant est refusé par défaut. `--force` permet explicitement
sa reconstruction sans contourner les contrôles d'intégrité. En cas d'erreur,
le temporaire créé par cette exécution est nettoyé et l'éventuel fichier final
précédent est préservé. Le pipeline ne remplace jamais une entrée source.

La CLI affiche un résumé JSON déterministe : `year`, `output`, `rows`, `resolved`,
`unresolved`, `resolved_COM`, `resolved_ARM`, `resolved_COMD` et `resolved_COMA`.
Les compteurs sont calculés pendant l'exécution ; les comptes réels suivants ne
sont pas codés comme des règles métier.

## Référence pour la validation manuelle ultérieure

L'audit annuel DVF/COG a établi les comptes attendus ci-dessous. Ils constituent
la référence du futur canari réel et ne sont pas les résultats d'une exécution
réelle de ce nouveau pipeline pendant son implémentation.

| Année | Lignes | Résolues | Unresolved | COM | ARM | COMD | COMA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021 | 745 132 | 744 726 | 406 | 690 261 | 54 463 | 2 | 0 |
| 2022 | 725 683 | 725 261 | 422 | 668 489 | 56 738 | 34 | 0 |
| 2023 | 577 260 | 576 922 | 338 | 531 009 | 45 903 | 10 | 0 |
| 2024 | 521 169 | 520 880 | 289 | 479 709 | 41 108 | 63 | 0 |
| 2025 | 591 274 | 591 043 | 231 | 543 369 | 47 621 | 53 | 0 |
| **Total** | **3 160 518** | **3 158 832** | **1 686** | **2 912 837** | **245 833** | **162** | **0** |

La couverture attendue est de **99,946654314261 %**. Les **3 160 518 mutations**
doivent toutes être conservées, y compris les **1 686 unresolved**. Un code
historiquement connu mais absent du référentiel annuel courant reste unresolved.
