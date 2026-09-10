# Normalisation du DVF brut

La source est le fichier DVF brut produit par la DGFiP, acquis depuis data.gouv.fr
et conservé dans une archive ZIP locale. La normalisation transforme sa structure
pour préparer les traitements suivants. Son grain reste **une ligne source DVF** :
chaque ligne de données produit exactement une ligne normalisée, dans le même ordre.
Il n'y a ni suppression, déduplication, agrégation, sélection de logement,
qualification métier, correction d'outlier, ni géocodage dans cette couche.

## Entrée, provenance et schéma source

Seules les années configurées dans `configs/data.yaml` sont acceptées : 2021 à
2025. Avant de transformer les données, le module vérifie la présence du ZIP,
du manifest local et de l'entrée correspondant à l'année, la correspondance du
nom de fichier et l'égalité entre le SHA-256 réel du ZIP et celui du manifest.
Une divergence interrompt le traitement sans requête réseau.

Le schéma source est défini explicitement dans `normalize.py` à partir des
43 colonnes du fichier DGFiP audité. Leurs noms exacts et leur ordre sont
contrôlés. Une colonne absente, dupliquée ou inattendue, un changement d'ordre,
ou une ligne dont le nombre de champs diffère du schéma provoque une erreur
explicite. Une donnée impossible à convertir provoque également une erreur ;
aucune ligne n'est éliminée pour poursuivre silencieusement la normalisation.

## Identifiant synthétique de mutation

Le brut DGFiP ne fournit pas d'identifiant stable de mutation. `id_mutation` est
créé en parcourant les lignes dans leur ordre source et en comparant la paire
`(date_mutation, valeur_fonciere)` à celle de la ligne précédente. Une différence
ouvre un nouveau groupe, numéroté `YYYY-N` à partir de `YYYY-1`.

Les chaînes sont nettoyées avec `strip`. La date est validée strictement au
format source `JJ/MM/AAAA`, puis convertie en `YYYY-MM-DD`. Le montant est comparé
exactement avec `Decimal`, après conversion de la virgule décimale : `100,00` et
`100` sont égaux. Un montant vide est distinct de zéro ; zéro reste une valeur
numérique. Aucun float binaire n'intervient dans la décision de frontière.

Les groupes sont **contigus** : si une paire réapparaît après une autre paire,
elle reçoit un nouvel identifiant. Un changement de `numero_disposition` seul
ne crée pas de groupe. Inversement, une date ou un montant différent crée un
nouveau groupe même si `numero_disposition` ne change pas.

> id_mutation is synthetic, snapshot-local and non-stable across DVF releases.
> It is intended only to group contiguous source rows.

Cette heuristique ne certifie pas l'identité juridique des mutations : deux
mutations distinctes adjacentes peuvent partager une date et un montant.
`numero_disposition` désigne une sous-partie juridique et n'est pas un identifiant
global. Ses zéros initiaux sont conservés.

## Identifiant de parcelle

Les composants sont traités comme des chaînes, après `strip`, puis complétés
à gauche par des zéros :

- Si le département commence par `97`, `code_commune` est le département suivi
  du code commune source complété à deux caractères.
- Sinon, `code_commune` est le département complété à deux caractères suivi du
  code commune source complété à trois caractères. Les départements `2A` et `2B`
  conservent leur représentation alphanumérique.
- `prefixe_section` est complété à trois caractères ; une valeur vide devient
  `000`.
- `section` est complétée à deux caractères et `numero_plan` à quatre caractères.
- `id_parcelle` concatène `code_commune`, `prefixe_section`, `section` et
  `numero_plan`, pour une longueur totale de 14 caractères.

Un composant nécessaire absent, une valeur invalide ou une longueur incompatible
provoque une erreur. Aucune valeur n'est tronquée pour fabriquer un identifiant.
Ces contrôles de format ne constituent pas une validation contre le cadastre.

## Colonnes et types Parquet

Le fichier contient uniquement les 23 colonnes suivantes :

| Colonnes | Type Arrow / Parquet |
| --- | --- |
| `source_year` | `int32` |
| `source_row_number` | `int64` |
| `id_mutation`, `date_mutation`, `numero_disposition`, `nature_mutation` | `string` |
| `valeur_fonciere` | `float64`, nullable |
| `code_postal`, `nom_commune`, `code_departement`, `code_commune` | `string` |
| `prefixe_section`, `section`, `numero_plan`, `id_parcelle` | `string` |
| `nombre_lots`, `code_type_local` | `int64`, nullables |
| `type_local` | `string` |
| `surface_reelle_bati` | `float64`, nullable |
| `nombre_pieces_principales` | `int64`, nullable |
| `nature_culture`, `nature_culture_speciale` | `string` |
| `surface_terrain` | `float64`, nullable |

`source_row_number` commence à 1, hors en-tête. Les nombres absents deviennent
`null` et les zéros restent des zéros. Les chaînes vides restent des chaînes vides,
sauf les composants de parcelle soumis aux règles ci-dessus. Les montants et
surfaces sont convertis en `float64` après les comparaisons exactes servant au
regroupement. Une valeur non numérique, non finie, ou dont la conversion produit
un infini ou réduit un nombre non nul à zéro est rejetée explicitement. Ces
contrôles sont techniques et ne fixent aucun seuil économique.

Les champs d'adresse détaillés (`No voie`, `Voie`, `Type de voie`, etc.) et les
autres colonnes source inutiles ne sont pas propagés. Les coordonnées ne sont
pas présentes dans cette sortie ; aucune longitude ou latitude n'est inventée.

## Lecture par batches et écriture atomique

Le fichier est lu directement dans le ZIP, sans extraction permanente et sans
charger l'année complète en mémoire. Le nombre de lignes par batch est configuré
avec `dvf.normalization.batch_size`. La dernière paire canonique date/montant et
le compteur de groupes sont conservés entre les batches : une coupure de batch
ne crée ni ne supprime une frontière de mutation.

Un `ParquetWriter` écrit progressivement dans un fichier temporaire `.part`, dans
le même répertoire que la destination. Après lecture complète, le schéma et le
nombre de lignes du Parquet sont contrôlés avant le remplacement atomique par
`os.replace`. En cas d'échec, le fichier temporaire est nettoyé et un fichier final
préexistant reste intact.

Le répertoire est défini par `dvf.normalization.output_directory`, avec un chemin
relatif au projet. La destination pour 2025 est
`data/interim/dvf/normalized/dvf_2025.parquet`. Les données brutes, le manifest et
les fichiers intermédiaires restent hors Git.

## Exécution explicite et étapes suivantes

Depuis la racine du projet, la commande suivante transforme uniquement l'année
demandée, en utilisant les fichiers locaux déjà acquis :

```bash
python -m real_estate.data.normalize --year 2025
```

Un fichier final existant entraîne un refus par défaut. Son remplacement doit
être demandé explicitement :

```bash
python -m real_estate.data.normalize --year 2025 --force
```

L'import du module ne déclenche aucun traitement. Le parcours prévu est :

```text
raw -> normalized -> qualification V1 -> enrichment -> ML
```

La qualification V1 appartient à `clean.py` et reste séparée de la normalisation.
L'interface avec cette couche devra être adaptée explicitement avant son
branchement, notamment pour les coordonnées absentes du brut. Cette étape ne
modifie ni `clean.py` ni `validate.py` et n'implémente pas les étapes suivantes.
