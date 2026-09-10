# Sources de données

## DVF — Demandes de valeurs foncières

DVF est la source officielle des transactions foncières retenue pour cette reconstruction. Elle est produite par la Direction générale des finances publiques (DGFiP) et publiée sur data.gouv.fr. La période retenue est 2021–2025 : elle correspond aux cinq millésimes officiellement publiés dans l'état courant de la source lors de cette configuration.

- Page officielle : [Demandes de valeurs foncières](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres).
- Licence annoncée par la page officielle : Licence Ouverte / Open Licence 2.0.
- Les données brutes ne sont jamais versionnées dans Git.
- Après une acquisition réelle, `data/raw/dvf/manifest.json` conserve localement l'URL source configurée (`source_url`), l'URL finale après redirection (`final_url`), le nom du fichier (`filename`), sa taille en octets (`bytes`), son SHA-256 calculé localement (`sha256`) et la date UTC de téléchargement (`downloaded_at`). Ce manifest identifie les octets effectivement acquis ; il n'est pas versionné par Git.

Les URL de ressources et les paramètres d'acquisition sont définis explicitement dans `configs/data.yaml`. Les ressources data.gouv.fr peuvent être remplacées lors des mises à jour semestrielles. Les URL `/api/1/datasets/r/{uuid}` redirigent vers la version courante d'une ressource et ne constituent pas des snapshots immuables. La traçabilité d'une acquisition réelle repose sur le manifest, notamment `final_url` et `sha256`. Le projet n'invente ni date de snapshot ni checksum officiel.

Les données DVF peuvent contenir des données à caractère personnel. Leur réutilisation ne doit pas permettre la ré-identification indirecte des personnes ni l'indexation de ces données par des moteurs de recherche externes.

## Sources à documenter ultérieurement

| Source prévue | Usage envisagé | État |
| --- | --- | --- |
| BPE — Base permanente des équipements | Caractéristiques de l'environnement local | Source exacte et millésime à vérifier |
| Référentiel des communes | Identification des communes et rapprochement des données | Référentiel et version à sélectionner et vérifier |
