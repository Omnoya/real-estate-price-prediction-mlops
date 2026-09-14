# Contrat du référentiel géographique annuel COG

Le référentiel utilise le Code officiel géographique (COG) INSEE au 1er janvier
du millésime concerné. Chaque année DVF doit utiliser le COG du même millésime.
Cette couche construit uniquement un référentiel annuel : elle ne lit pas les
Parquet DVF, ne réalise aucune jointure et n'harmonise pas les codes vers une
géographie plus récente.

## Sources et provenance

L'acquisition reste la responsabilité de `real_estate.data.communes`.
`real_estate.data.geography` lit uniquement le CSV courant explicitement configuré
dans `cog.geography.current_commune_files`, directement dans le ZIP annuel :

| Année | Archive sous `data/raw/cog/` | CSV courant |
| --- | --- | --- |
| 2021 | `cog_2021.zip` | `commune2021.csv` |
| 2022 | `cog_2022.zip` | `commune_2022.csv` |
| 2023 | `cog_2023.zip` | `v_commune_2023.csv` |
| 2024 | `cog_2024.zip` | `v_commune_2024.csv` |
| 2025 | `cog_2025.zip` | `v_commune_2025.csv` |

Avant lecture du CSV, le module vérifie l'existence de l'archive et de son entrée
annuelle dans `data/raw/cog/manifest.json`. Le nom `filename`, l'URL configurée
`source_url`, la taille `bytes` et le SHA-256 recalculé doivent correspondre.
Ces contrôles sont locaux et ne déclenchent aucune requête réseau. Les URL ne
sont pas considérées comme des snapshots immuables ; le manifest et le SHA-256
identifient les octets acquis. Le manifest et l'archive ne sont pas modifiés.

Le CSV courant possède les 12 colonnes suivantes, avec des codes lus comme
chaînes pour conserver les zéros initiaux :

```text
TYPECOM, COM, REG, DEP, CTCD, ARR, TNCC, NCC, NCCENR, LIBELLE, CAN, COMPARENT
```

Les fichiers historiques et les journaux de mouvements ne sont jamais lus par
ce module. Ils ne servent jamais à remapper automatiquement un code.

## Résolution hiérarchique

Le grain de sortie est une ligne par `COM` distinct du COG courant, renommé
`source_code_commune`. La sélection suit exactement cette priorité :

1. `COM` : retenir l'unique commune portant le code source ; elle est aussi la
   commune canonique et son `parent_commune_code` de sortie est null.
2. `ARM` : retenir l'arrondissement municipal et sa commune parente `COMPARENT`.
3. `COMD` : retenir la commune déléguée et sa commune parente `COMPARENT`.
4. `COMA` : retenir la commune associée et sa commune parente `COMPARENT`.

Pour ARM, COMD et COMA, `COMPARENT` doit désigner exactement une ligne
`TYPECOM=COM` dans le COG courant du même millésime. Le libellé canonique, la
région et le département proviennent de cette COM canonique. Les éventuelles
valeurs REG et DEP de la ligne source ne les remplacent pas.

Un code peut avoir une ligne COM et une ligne COMD. La COM est alors prioritaire,
sans créer d'ambiguïté. Le code source seul ne permet pas de deviner si une
future mutation concernait historiquement la commune déléguée. Le code source
reste conservé, y compris lorsqu'il désigne un ARM ou une COMD/COMA dont la
commune canonique porte un autre code. Aucun libellé n'est utilisé comme clé.

## Validations

Le module refuse un schéma incompatible, une structure CSV invalide, un COM vide
ou de longueur différente de cinq caractères, un TYPECOM autre que COM, ARM,
COMD ou COMA et toute duplication de `(TYPECOM, COM)`. Il ne déduplique pas les
entrées source.

Chaque ARM, COMD et COMA doit avoir un COMPARENT renseigné de cinq caractères,
présent dans l'index et correspondant à exactement une COM. Les validations
portent aussi sur les lignes qui ne seront pas sélectionnées, notamment une
COMD masquée par la priorité COM. Une structure source incompatible provoque une
erreur explicite plutôt qu'une résolution partielle.

## Schéma Parquet

Les colonnes sont écrites dans l'ordre suivant. Les huit premières sont des
chaînes Arrow `string`, la dernière est un entier Arrow `int32`. Seul
`parent_commune_code` est nullable.

| Champ | Définition |
| --- | --- |
| `source_code_commune` | COM source inchangé |
| `resolved_geo_type` | TYPECOM retenu selon COM → ARM → COMD → COMA |
| `source_geo_label` | LIBELLE de la ligne retenue |
| `parent_commune_code` | Null pour COM ; COMPARENT pour ARM/COMD/COMA |
| `canonical_commune_code` | Code source pour COM ; COMPARENT pour ARM/COMD/COMA |
| `canonical_commune_label` | LIBELLE de l'unique COM canonique |
| `region_code` | REG de la COM canonique |
| `department_code` | DEP de la COM canonique |
| `cog_year` | Millésime demandé |

Avant publication, le Parquet doit être non vide, respecter ce schéma et ne
contenir qu'une ligne par code source. Les codes source et canoniques doivent
avoir cinq caractères ; les types doivent appartenir aux quatre catégories
autorisées ; les libellés canoniques, régions et départements doivent être
renseignés. Le millésime est uniforme et égal à celui demandé.

Le parent est null exactement pour COM, dont le code canonique égale le code
source. Pour ARM/COMD/COMA, le parent est renseigné et égal au code canonique.

## Exécution et publication

L'API sépare configuration, provenance, lecture CSV, validation, indexation,
résolution et publication. La mémoire contient uniquement le COG annuel,
environ 37 000 lignes, sans charger de données DVF. Les observations sont
ordonnées par `source_code_commune` lexicographiquement pour une sortie
déterministe.

```bash
python -m real_estate.data.geography --year 2025
python -m real_estate.data.geography --year 2025 --force
```

Seules les années configurées sont acceptées. La destination est
`data/interim/cog/geography/cog_geography_YYYY.parquet`, avec un répertoire
configurable et relatif au projet. `data/interim` reste hors Git.

Le module écrit un fichier temporaire `.part`, le valide, synchronise son
contenu avec `fsync`, puis utilise `os.replace` pour le publier atomiquement.
Tout échec nettoie le temporaire et préserve un fichier final existant. Un
fichier final existant est refusé par défaut ; `--force` autorise explicitement
sa reconstruction. La CLI affiche le millésime, la destination, le nombre de
codes source et les comptes COM/ARM/COMD/COMA, sans lister toutes les lignes.

## Références d'audit et validation manuelle ultérieure

Les comptes attendus suivants décrivent le référentiel après priorité COM,
d'après l'audit des archives locales 2021–2025. Ils ne sont pas des règles
codées en dur et les tests unitaires n'utilisent pas les archives réelles.

| Millésime | Codes source | COM | ARM | COMD | COMA |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2021 | 37 122 | 34 965 | 45 | 1 595 | 517 |
| 2022 | 37 026 | 34 955 | 45 | 1 518 | 508 |
| 2023 | 36 993 | 34 945 | 45 | 1 511 | 492 |
| 2024 | 36 971 | 34 935 | 45 | 1 508 | 483 |
| 2025 | 36 949 | 34 875 | 45 | 1 553 | 476 |

L'audit géographique DVF/COG 2021–2025 a résolu 3 158 832 mutations sur
3 160 518, soit une couverture de 99,946654314261 %, avec zéro ambiguïté finale.
Les 1 686 mutations restantes sont restées `unresolved`, même lorsqu'un code
était retrouvé dans une source historique. Ces nombres décrivent l'audit déjà
réalisé ; ils ne constituent pas une jointure effectuée par ce module.

Lors de l'enrichissement DVF futur, les codes absents du COG courant du même
millésime resteront `unresolved`. `DVF.code_commune` restera inchangé. La BPE
et toute harmonisation historique sont hors du périmètre de ce référentiel.
