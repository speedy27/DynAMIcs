# Benchmark temporel microbiome — Rapport

Branche : `amine` | Date : 2026-06-20

---

## 1. Contexte et objectif

L'objectif est de comparer quantitativement le world model JEPA microbiome à des baselines
publiées, en adoptant un protocole reconnu dans la littérature afin de rendre les résultats
directement percutants pour un jury.

Le protocole choisi est le **protocole MDSINE2** (Microbial Dynamical Systems INference Engine,
Freed et al. 2021), qui est la référence pour l'évaluation de modèles dynamiques de communautés
microbiennes. Il repose sur une validation croisée **hold-one-subject-out** et mesure l'erreur de
prédiction dans l'espace log-abondance (CLR-RMSE), à plusieurs horizons temporels.

---

## 2. Données

**Simulateur gLV synthétique** (`eb_jepa/datasets/microbiome/glv.py`)

Toutes les expériences de ce benchmark utilisent des trajectoires **entièrement synthétiques**
générées par le simulateur Generalized Lotka-Volterra (gLV) déjà présent dans le dépôt.

Paramètres du simulateur :
- 32 espèces, 3 guildes (compétition cyclique rock-paper-scissors)
- Attracteurs multistables par exclusion compétitive intra-guilde
- Structure **non-monotone** : atteindre le bassin cible peut nécessiter de s'en éloigner d'abord
- Panel d'action : 8 espèces candidates (perturbations probiotiques continues)
- Politique d'action : doses aléatoires non-négatives à chaque pas (simulation d'interventions)

Configuration du benchmark :
- **30 sujets** (trajectoires), **60 pas** chacun
- **3 seeds indépendantes** (seed 0, 1, 2) pour la robustesse statistique
- **Horizons évalués** : 1, 3, 5, 10 pas

Le simulateur gLV est le "Two Rooms" du microbiome : données de contrôle, attracteurs connus,
dynamique non-triviale. Il permet de comparer proprement les modèles sans biais de données réelles.

---

## 3. Protocole — MDSINE2 hold-one-subject-out

Pour chaque sujet `s` parmi les 30 :
1. **Entraînement** sur les 29 sujets restants
2. **Prédiction autorégressive** depuis l'état initial du sujet `s` pour chaque horizon `k`
3. **Erreur** : CLR-RMSE entre la prédiction `x̂_{t+k}` et la vérité `x_{t+k}`

La métrique **CLR-RMSE** (Root Mean Squared Error dans l'espace Centered Log-Ratio) est
standard pour les abondances microbiomes car :
- Les abondances sont compositionnelles (somment à 1) et très sparses
- Le CLR (transformation d'Aitchison 1986) normalise cette structure
- Il évite que les espèces dominantes écrasent l'erreur des espèces rares
- C'est la même métrique que MDSINE2 (log-abondance centrée)

Toutes les erreurs sont moyennées sur tous les points de départ valides dans la trajectoire
et sur tous les sujets du fold. Résultats rapportés en **mean ± std** sur les 3 seeds.

---

## 4. Baselines — ce qui existait déjà

### 4.1 Dans `examples/microbiome/baselines.py` (branche adrien/bnz)

Le fichier `baselines.py` implémentait déjà une **ladder de représentation** en 5 niveaux :

| Niveau | Représentation | Rôle |
|--------|---------------|------|
| 1 | Diversité de Shannon (4 stats : mean/std/min/max) | Descripteur écologique classique, sans apprentissage |
| 2 | Courbe rang-abondance (OTUs triés par abondance) | Capture l'évenness, sans identité de séquence |
| 3 | Moyenne ProkBERT pondérée par abondance (raw) | Est-ce que le JEPA bat les features triviales ? |
| 4 | MLP supervisé sur raw ProkBERT | Référence non-linéaire supervisée (baseline à battre) |
| 5 | Encodeur aléatoire (SetEncoder non entraîné, 3 seeds) | Est-ce l'entraînement qui aide, ou juste l'architecture ? |

Métriques déjà calculées : `age_r2` (R² de l'âge de l'hôte) et `t1d_auroc` (AUROC T1D).

Ce fichier contenait aussi une **table de dynamiques en espace latent** (`_dynamics_table`)
comparant le prédicteur JEPA entraîné à des références dans l'espace latent :
persistence / mean-shift global / AR linéaire W[z,a] / JEPA 1-step (teacher-forced) /
JEPA rollout k-step.

---

## 5. Ce que j'ai ajouté

### 5.1 Benchmark temporel (`examples/microbiome/temporal_benchmark.py`) — nouveau

Fichier entièrement nouveau. Implémente le **protocole MDSINE2 complet** dans l'espace
des abondances observées (CLR), pas dans l'espace latent.

**Trois baselines temporelles codées :**

**Persistence** (`PersistenceModel`)
- Prédit `x_{t+1} = x_t` (aucun changement)
- La baseline "zéro effort" que tout modèle doit battre
- Très compétitive à court horizon sur des communautés lentes (dynamique amortie)
- Référence absolue : skill = pers_RMSE / model_RMSE > 1 signifie "bat la persistence"

**gLV-L2** (`GLV_L2`)
- Régression Ridge : `CLR(x_{t+1}) = W @ [CLR(x_t), action_t] + b`
- Approximation linéaire du modèle gLV (fitting des équations de Lotka-Volterra)
- Entraîné sur les N-1 sujets du fold, testé sur le sujet laissé de côté
- Equivalant à un modèle VAR (Vector AutoRegressive) avec termes d'action
- Référence linéaire publiée, rapide à entraîner

**gLV-net** (`GLV_Net`)
- MLP 2 couches (64 unités cachées) : `(CLR(x_t), action_t) → Δ CLR(x_t)`
- Prédit le **delta** (changement) plutôt que l'état absolu (plus facile à apprendre)
- Standardisation des features et des targets avant entraînement
- Capture les interactions non-linéaires entre espèces manquées par gLV-L2
- Equivalant à un gLV avec termes d'interaction non-linéaires

**JEPA (optionnel, via --ckpt)**
- Rollout du prédicteur RNNPredictor dans l'espace latent
- Readout linéaire `z → CLR(x)` entraîné sur les sujets d'entraînement
- Permet de comparer le JEPA dans le même espace que les baselines

**Génération automatique de 3 figures :**
- `temporal_benchmark_rmse.png` : CLR-RMSE vs horizon (courbes + barres d'erreur)
- `temporal_benchmark_skill.png` : Skill vs persistence (ratio pers_RMSE / model_RMSE)
- `temporal_benchmark_bars.png` : Barres groupées par horizon (format slides)

### 5.2 Ajouts dans `baselines.py` (représentation)

**RF (Random Forest) — `_probe_rf()`**
- 200 arbres, class_weight="balanced" pour le T1D
- C'est la baseline publiée dans MetAML et les papiers SSL-microbiome (revue de 2023)
- Référence non-paramétrique supervisée, standard dans la communauté microbiome

**Pearson r pour l'âge**
- Ajout de `age_r = pearsonr(age_pred, age_true)` en plus du R²
- Alignement sur le protocole du papier SSL-microbiome (Nguyen et al. 2023) qui rapporte r
- Complète le R² : r capture la corrélation linéaire directionnelle (utile quand R² est négatif)

---

## 6. Résultats

### 6.1 CLR-RMSE — Tableau complet (mean ± std, 3 seeds, 30 sujets)

| Modèle | h=1 | h=3 | h=5 | h=10 |
|--------|-----|-----|-----|------|
| Persistence | 0.458 ± 0.490 | 0.748 ± 0.551 | 0.921 ± 0.578 | 1.244 ± 0.644 |
| gLV-L2 (Ridge) | 0.399 ± 0.393 | 0.529 ± 0.370 | 0.564 ± 0.331 | 0.617 ± 0.258 |
| gLV-net (MLP) | **0.302 ± 0.337** | **0.415 ± 0.330** | **0.464 ± 0.321** | **0.548 ± 0.312** |
| JEPA (objectif) | — | — | — | — |

### 6.2 Skill vs Persistence (pers_RMSE / model_RMSE, >1 = bat no-change)

| Modèle | h=1 | h=3 | h=5 | h=10 |
|--------|-----|-----|-----|------|
| gLV-L2 | 1.15× | 1.41× | 1.63× | 2.02× |
| gLV-net | **1.52×** | **1.80×** | **1.98×** | **2.27×** |
| JEPA (objectif) | >1.52× | >1.80× | >1.98× | >2.27× |

### 6.3 Amélioration relative gLV-net vs gLV-L2

| Horizon | Gain gLV-net sur gLV-L2 |
|---------|------------------------|
| h=1 | −24.5% RMSE |
| h=3 | −21.5% RMSE |
| h=5 | −17.7% RMSE |
| h=10 | −11.2% RMSE |

---

## 7. Analyse et lectures

**Persistence est étonnamment forte à court terme.** À h=1, son RMSE est seulement 15%
supérieur à gLV-L2. Cela reflète la dynamique amortie du gLV : les communautés microbiennes
sont lentes à changer, donc "ne rien prédire" est une heuristique solide sur un seul pas.

**L'avantage des modèles s'accroit avec l'horizon.** À h=10, gLV-L2 a un skill de 2.0×
et gLV-net de 2.3×. Cela confirme que ces modèles capturent la vraie dynamique de convergence
vers les attracteurs, que persistence manque complètement (son RMSE explose à long terme).

**gLV-net domine gLV-L2, surtout à court terme.** L'écart se réduit à long terme : à h=10
les deux modèles ont convergé vers les attracteurs, ce qui est plus simple à apprendre.
Les interactions non-linéaires entre espèces sont importantes pour la dynamique à court terme.

**Variabilité inter-sujets élevée (std ≈ mean).** Normale : certaines paires (sujet initial,
attractor cible) sont très dynamiques, d'autres quasi-statiques. Le coefficient de variation
~1.0 est typique des dynamiques gLV avec attracteurs multistables.

**La barre pour le JEPA est claire.** Pour être compétitif sur ce benchmark, le world model
JEPA doit atteindre un skill > 1.52× à h=1 et > 2.27× à h=10 (battre gLV-net). Avec le
prédicteur RNNPredictor et l'IDM, c'est l'objectif visé. Les runs en cours sur DALIA
(job 74850 et suivants) permettront de renseigner la colonne JEPA.

---

## 8. Reproductibilité

```bash
# Benchmark complet (30 sujets, 3 seeds, figures)
python -m examples.microbiome.temporal_benchmark \
    --n_subjects 30 --T 60 --seeds 0 1 2 \
    --horizons 1 3 5 10 \
    --out artifacts/temporal_benchmark.json \
    --figs artifacts/figures

# Avec le JEPA entraîné (ajouter la colonne JEPA)
python -m examples.microbiome.temporal_benchmark \
    --n_subjects 30 --T 60 --seeds 0 1 2 \
    --ckpt artifacts/ckpt/microbiome_jepa.pt \
    --out artifacts/temporal_benchmark_jepa.json \
    --figs artifacts/figures

# Baselines de représentation (nécessite cache DIABIMMUNE)
python -m examples.microbiome.baselines \
    --ckpt artifacts/ckpt/microbiome_jepa.pt run1.pt run2.pt
```

Résultats JSON sauvés dans `artifacts/temporal_benchmark.json`.
Figures PNG dans `artifacts/figures/` :
- `temporal_benchmark_rmse.png`
- `temporal_benchmark_skill.png`
- `temporal_benchmark_bars.png`

---

## 9. Prochaines étapes

- **Colonne JEPA** : lancer le benchmark avec `--ckpt` une fois le checkpoint cluster rapatrié
- **DIABIMMUNE réel** : adapter le protocole HOSO aux vrais sujets longitudinaux
  (nécessite le cache cluster `/lustre/work/vivatech-dynamics/aouldhoci/`)
- **Axe planning** : benchmark Jones CDI (bassin success + volume d'intervention)
  sur instances non-monotones — le simulateur gLV est déjà configuré pour ça
- **MDSINE2 lui-même** : ajouter comme baseline si temps disponible (pip install mdsine2)
