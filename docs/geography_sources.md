# Sources géographiques

## INSEE — Code officiel géographique (COG)

Le projet retient les archives CSV du COG publiées par l'INSEE pour les millésimes
2021, 2022, 2023, 2024 et 2025. Chaque millésime décrit la géographie au 1er janvier
de l'année correspondante. Un COG annuel est prévu pour chaque année DVF ; aucune
harmonisation vers une géographie courante n'est réalisée à cette étape.

La [page générale officielle du COG](https://www.insee.fr/fr/information/2560452)
et les URL des archives sont configurées dans la section `cog` de
`configs/data.yaml` :

| Millésime | Archive CSV configurée |
| --- | --- |
| 2021 | [cog_ensemble_2021_csv.zip](https://www.insee.fr/fr/statistiques/fichier/5057840/cog_ensemble_2021_csv.zip) |
| 2022 | [cog_ensemble_2022_csv.zip](https://www.insee.fr/fr/statistiques/fichier/6051727/cog_ensemble_2022_csv.zip) |
| 2023 | [cog_ensemble_2023_csv.zip](https://www.insee.fr/fr/statistiques/fichier/6800675/cog_ensemble_2023_csv.zip) |
| 2024 | [cog_ensemble_2024_csv.zip](https://www.insee.fr/fr/statistiques/fichier/7766585/cog_ensemble_2024_csv.zip) |
| 2025 | [cog_ensemble_2025_csv.zip](https://www.insee.fr/fr/statistiques/fichier/8377162/cog_ensemble_2025_csv.zip) |

Ces URL configurées ne sont pas présentées comme des snapshots immuables. Leur
vérification réseau et la première acquisition réelle seront réalisées séparément
après revue du code ; aucun téléchargement réel n'est nécessaire aux tests.

## Acquisition et provenance locale

Le module `real_estate.data.communes` stocke uniquement les archives sous
`data/raw/cog/cog_YYYY.zip`. Il ne les extrait pas de manière permanente et ne
produit aucun CSV dans `interim` ou `processed`. Le futur lecteur pourra ouvrir
les CSV directement dans le ZIP. Les archives et le manifest local restent hors
Git, comme l'ensemble des données sous `data/raw`.

L'acquisition est déclenchée explicitement :

```bash
python -m real_estate.data.communes --year 2025
python -m real_estate.data.communes --year 2025 --force
python -m real_estate.data.communes --year 2021 2022 2023 2024 2025
```

Après acquisition, `data/raw/cog/manifest.json` contient une entrée par année dans
`downloads`, indexée par l'année sous forme de chaîne. Chaque entrée conserve :

- `year` : millésime entier ;
- `source_url` : URL de ressource configurée ;
- `final_url` : URL finale après les éventuelles redirections ;
- `filename` : nom local de l'archive ;
- `bytes` : taille effectivement téléchargée ;
- `sha256` : empreinte SHA-256 calculée localement pendant le téléchargement ;
- `downloaded_at` : date de téléchargement UTC au format ISO-8601.

Le manifest et le SHA-256 identifient les octets acquis. Aucun checksum officiel
ni date de snapshot supplémentaire n'est inventé. Si le fichier existant et son
entrée de manifest concordent, notamment pour `filename`, `source_url` et le
SHA-256 recalculé, aucune requête réseau n'est effectuée. Un état partiel ou
incohérent provoque une erreur ; `--force` demande explicitement un nouveau
téléchargement.

Les requêtes utilisent le streaming, les timeouts de connexion et de lecture et
un nombre borné de tentatives en cas d'erreur réseau ou HTTP. Chaque tentative
écrit un nouveau `.part`, calcule le SHA-256 et vérifie la taille non nulle, le
`Content-Length` éventuel et la structure ZIP, sans extraire ses membres.
L'archive et le JSON sont vidés avec `fsync` avant leurs remplacements atomiques.
Seules les sessions HTTP créées par le module sont fermées par celui-ci.

En cas d'échec de publication du manifest, l'archive précédente est restaurée
à partir d'un lien temporaire ; les temporaires sont nettoyés. Les deux fichiers
ne constituent cependant pas une transaction atomique en cas d'arrêt brutal
du processus entre leurs remplacements : un tel état est détecté au prochain
appel. Ces acquisitions locales doivent être exécutées une à la fois.
`--force` ne réinitialise jamais un manifest JSON illisible ; celui-ci provoque
une erreur avant toute requête afin de préserver la provenance des autres années.

## Référentiel et jointure à étudier

Cette étape ne filtre pas `TYPECOM`, ne déduplique aucune ligne et ne fixe aucune
clé de jointure. Le champ `COM` ne doit pas être supposé unique dans le fichier
complet sans audit : une commune et une commune déléguée peuvent notamment
partager le même code. Les catégories à examiner incluent :

| `TYPECOM` | Catégorie |
| --- | --- |
| `COM` | Commune |
| `COMA` | Commune associée |
| `COMD` | Commune déléguée |
| `ARM` | Arrondissement municipal |

Les champs disponibles, notamment `TYPECOM`, `COM`, `REG`, `DEP`, `TNCC`, `NCC`,
`NCCENR`, `LIBELLE` et `COMPARENT`, devront être audités pour chaque millésime
avant de construire un référentiel final. Aucune jointure avec DVF ni modification
de ses `code_commune` n'est effectuée ici. La BPE viendra après la stabilisation
du référentiel géographique.
