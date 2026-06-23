# Monitoring scientifique EVE — site_004

- Généré : 2026-06-23T13:23:36+00:00
- Statut : `warn`

## Résumé

| Indicateur | Valeur |
| --- | --- |
| Points de décision | 1479 |
| Période | 2015-07-30 → 2026-06-17 |
| Points en saison | 879 |
| Points éligibles | 477 |
| Candidats | 108 |
| Épisodes | 32 |
| Taux candidat / éligible | 0.226 |
| Part observations pauvres en saison | 0.457 |

## Alertes

- Plus de 40 % des observations en saison sont de qualité satellite pauvre.
- Au moins un épisode dure 45 jours ou plus : vérifier un éventuel chaînage.

## Invariants

| Contrôle | Valeur |
| --- | --- |
| duplicate_decision_point_ids | 0 |
| candidate_without_episode | 0 |
| non_candidate_with_episode | 0 |
| poor_satellite_candidate | 0 |
| ndmi_percentile_out_of_bounds | 0 |
| episode_count_mismatch | 0 |

## Répartition des règles

| Règle | Nombre |
| --- | --- |
| R0 | 600 |
| R1 | 402 |
| R2 | 369 |
| R4 | 55 |
| R6 | 42 |
| R5 | 9 |
| R3 | 2 |

## Actions épisodes

| Action | Nombre |
| --- | --- |
| priority_field_check | 17 |
| review_raster | 12 |
| field_check | 3 |

## Épisodes les plus longs

| Épisode | Début | Fin | Durée | Points | Action | Hydro | VPD | NDMI pct min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| site_004_episode_20200606_13 | 2020-06-06 | 2020-07-31 | 55 | 14 | priority_field_check | very_dry | high | 5.882 |
| site_004_episode_20250401_29 | 2025-04-01 | 2025-05-16 | 45 | 11 | review_raster | normal | high | 0.000 |
| site_004_episode_20180707_07 | 2018-07-07 | 2018-08-03 | 27 | 8 | priority_field_check | very_dry | normal | 1.449 |
| site_004_episode_20200820_14 | 2020-08-20 | 2020-09-16 | 27 | 6 | priority_field_check | very_dry | normal | 5.714 |
| site_004_episode_20190907_11 | 2019-09-07 | 2019-09-30 | 23 | 7 | priority_field_check | dry | high | 1.754 |
| site_004_episode_20171010_06 | 2017-10-10 | 2017-10-30 | 20 | 6 | priority_field_check | very_dry | very_high | 0.000 |
| site_004_episode_20180821_08 | 2018-08-21 | 2018-09-10 | 20 | 5 | priority_field_check | very_dry | very_high | 2.985 |
| site_004_episode_20220512_21 | 2022-05-12 | 2022-05-29 | 17 | 4 | priority_field_check | very_dry | very_high | 4.651 |
| site_004_episode_20240630_27 | 2024-06-30 | 2024-07-15 | 15 | 2 | review_raster | normal | normal | 1.515 |
| site_004_episode_20250617_30 | 2025-06-17 | 2025-06-29 | 12 | 3 | priority_field_check | very_dry | very_high | 4.839 |

## Coûts théoriques

| Périmètre | Épisodes | Temps min | Temps max | Coût min | Coût max |
| --- | --- | --- | --- | --- | --- |
| Tous épisodes | 32 | 1320 | 2550 | 400.000 | 1400.000 |
| Priorité terrain | 17 | 1020 | 2040 | 340.000 | 1190.000 |
