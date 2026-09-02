# masking-ceiling

Companion code for "Geometric Ceilings on Time-Frequency Masking for Single-Channel Separation". The paper states the exact ceiling of the real-gain format, the orthogonal projection of the source onto the line spanned by the mixture, places an arbitrary estimator on a chain of four nested real-linear operator classes, and measures where a non-circular Gaussian-mixture prior lands under that ceiling on MUSDB18. This repository holds the measurements.

The separator itself is imported from [`gasm`](https://github.com/mbaelde/generative-audio-source-models), the thesis code, so what is being located with respect to the class is the reference implementation and not a paraphrase of it. Section and table numbers below refer to the article; each experiment section states which of its claims that experiment produces.

## From the article to the code

| Article | Script | Question it answers |
| --- | --- | --- |
| §6.3, §6.4 | `evaluation.py` | WOLA resynthesis, time-domain SDR and the exact ceiling of the class in closed form. |
| §6.4 | `ola_projection_check.py` | Does the resynthesis transport a coefficient inequality to a ratio of dB, or only approximate it? |
| §7.1 | `theta_distribution.py` | Does the ceiling recomputed from the energy-weighted quantiles of the phase disagreement cross-check the closed form? |
| §7.2, Table 2 | `report_ceiling.py --cascade` | What does each of the four orthogonal projections of the fixed-operator reading add? |
| §7.3, Table 3 | `ceiling_sweep.py`, `report_ceiling.py` | How does the deficit to the ceiling move with the number of components and the training volume? |
| §7.4, Table 4 | `ceiling_sweep.py` (full-covariance configuration) | Does the last class of the chain, full covariance across frequencies, close the gap the diagonal one leaves? |
| §7.5, Table 5 | `estimator_readout.py` | Do read-outs that are not means show the class witnesses the mean does not? |
| §7.6, Table 6 | `prior_calibration.py` | Is the deficit honest posterior variance or an artefact of fitting? |
| §7.6 | `calibration_cross_check.py` | Does the spectral deficit reconcile with the time-domain one by an identity? |
| §7.7, Table 7 | `law_diagnostic.py` | Do the isolated-source spectra satisfy the three assumptions the prior makes? |
| §7.8 | `offsupport_path.py` | Does the error away from the training support localise to the responsibility term rather than the regression term? |
| Fig. 1, Fig. 4 | `figures/geometry.py` | The matplotlib prototype of the geometry the closed-form ceiling and the operator hierarchy rest on. The figures of record are redrawn in TikZ inside the manuscript, so this script documents the construction rather than producing the published plates. |

## Layout

- `experiments/evaluation.py`, WOLA analysis and resynthesis, time-domain SDR, and the closed-form oracles of the real-gain class and its two bounded subclasses.
- `experiments/ceiling_sweep.py`, the corpus, the excerpt selection and the Gaussian-mixture fit and sweep the article's tables come from.
- `experiments/report_ceiling.py`, journal loading, the silence guard and the cascade reading.
- `experiments/ola_projection_check.py`, `theta_distribution.py`, `estimator_readout.py`, `law_diagnostic.py`, `prior_calibration.py`, `calibration_cross_check.py`, `offsupport_path.py`, one script per claim, each self-contained on top of the two layers above.
- `figures/geometry.py`, the matplotlib prototype of the two geometric figures, superseded by the TikZ versions in the manuscript.
- Runners: `run.sh`, `local.sh`, `daneel.sh`.

The modules import each other flat, which is why they sit in one directory: a script is run with `experiments/` as `sys.path[0]`, and `run.sh` is what groups them into a plan. Every script carries a `_self_check()` reachable with `--check` that runs before the corpus is touched, and takes its parameters from environment variables so one image runs every cell of a plan. Every script writes one JSON object per line on stdout, so a log is directly a dataframe.

## The experiments

`experiments/evaluation.py` is the measurement layer: WOLA analysis and resynthesis, time-domain SDR, and the exact ceiling of the class in closed form. `m*[f] = Re(s[f] conj(x[f])) / |x[f]|^2` with residual `|s|^2 sin^2(theta)`. It reports the two restrictions of the class next to it, `positive` and `clipped`, each with its own closed-form ceiling, because a comparison against one of them bounds no other. The IRM and the oracle Wiener filter sit about 6 dB below `m*` on this material, which is why nothing here is quoted against the IRM alone. One full frame is trimmed at each end inside the metrics: a modified spectrogram divided by the sum of squared analysis windows explodes where that sum decays, and an untrimmed oracle then reads below the mixture.

`experiments/ola_projection_check.py` settles what the resynthesis does to a coefficient error, §6.4 resting on it. The projection is orthogonal rather than oblique, `synthesize` normalising by the canonical dual window of a painless Gabor frame, and `S (I - A S) = 0` exactly. The frame is not tight at hop L/2, so the transport of the coefficient inequality to a ratio of dB is measured rather than assumed.

`experiments/theta_distribution.py` opens the ceiling up: energy-weighted quantiles of `|theta|` on real music, the share of source energy past a few angles, and the ceiling recomputed from that distribution as a cross-check against `spectral_ceiling`. No prior, no fit.

`experiments/ceiling_sweep.py` is the sweep the article's tables come from: fit a non-circular Gaussian mixture per source on stacked real and imaginary spectra, read out the posterior mean over component pairs, and score it against `m*` rather than against an oracle mask. It is the expensive state of the repository, hours per cell at `K = 32`, so the EM runs in chunks through scikit-learn's `warm_start` and checkpoints after each: a killed run resumes at the last chunk, and a fit stopped at ten iterations is extended to thirty by a later run at a higher `MAX_ITER` instead of being thrown away. The iteration count deliberately lives in the file and not in the checkpoint key.

`experiments/report_ceiling.py` turns those journals into the table, one row per (probe, source, K). The deciding column is `d(m*)`, the mean over test tracks of `sdr_dmgmm - sdr_best_real`, paired track by track because tracks differ in difficulty by more than the margin being measured. It drops tracks with a silent stem and says which, and `--cascade` reads the same oracle rows as the four-projection cascade of §7.2.

`experiments/estimator_readout.py` asks the class question of two read-outs that are not means, the conditional mean of the most probable component pair and one draw, the posterior mean being provably pulled back onto the line by the squared-error criterion itself. The prediction is signed: both must show the class witnesses the mean barely shows, phase rotation and gains outside `[0, 1]`, and both must score worse in SDR. A draw that stayed inside the class would move the blame from the criterion to the partition.

`experiments/law_diagnostic.py` looks at the data rather than at the fitted model, four statistics each testing one assumption the prior makes, each against the value the same statistic takes at the same sample size on synthetic data that satisfies it. That null column is the point: at 300 frames per cell a measured non-circularity of 0.05 is evidence of nothing. The rows that matter are conditional on the fitted model's own hard assignment, since the model never sees the marginal.

`experiments/prior_calibration.py` computes both halves of §7.6 on the same checkpoints, the excess the prior predicts against the excess it realises, and splits the predicted variance into a within-pair term, which a Gaussian scale mixture replaces wholesale, and a between-pair term, which it leaves alone. It is the gate on writing a Student EM: a deficit that is between-pair says the lever is components or conditioning and not heavier tails.

`experiments/calibration_cross_check.py` reconciles the spectral deficit with the time-domain one on one track, by an identity rather than an argument. Splitting both estimate and truth on the mixture makes the spectral error a sum with no cross term, whose first term is exactly what the closed form computes; whatever remains against the measured time-domain error is the overlap-add projection and nothing else.

`experiments/offsupport_path.py` tests §7.8's off-support claim, that the responsibility term rather than the regression term carries the sensitivity. Under a spectral tilt applied to both sources and renormalised in energy, the error should grow smoothly while the arg max of the responsibilities is stable and break where it switches. The statistic is the jump in error at switching steps against the jump at non-switching ones; a ratio near one refutes the analysis.

`figures/geometry.py` draws the two geometric figures, one bin of the complex plane and the ladder of block structures read as images of the unit circle. It is a PEP-723 script and carries its own dependencies.

## Running

```
uv run python experiments/report_ceiling.py --check                  # self-checks, no corpus
uv run python experiments/ola_projection_check.py
MUSDB_ROOT=/path/to/musdb18 uv run python experiments/theta_distribution.py
sh experiments/run.sh cond                                           # the conditioning probes
sh experiments/run.sh pilot                                          # the GO/NO-GO pilot, resumable
MAX_TRAIN_FRAMES=150000 sh experiments/local.sh pilot                # the same, PC perso, full MUSDB18
LOG=/scratch/logs/classes sh experiments/daneel.sh classes           # detached, in a container
uv run python experiments/report_ceiling.py scratch/logs/ceiling/*.jsonl
py -3.14 -m uv run figures/geometry.py figures
```

`experiments/run.sh` is the form meant for a real run: one plan per invocation, each cell skipping what its journal already holds and each EM fit restarting from its last checkpoint, so a killed run is relaunched with the same command. `LOG` needs a directory of its own per arm, cells being named by plan and skipped on their `.done` marker: a second arm run into the first one's directory is skipped whole and hands back a complete journal that is not the arm asked for. `MAX_TRAIN_FRAMES` is part of the checkpoint key and the MUSDB edition is not, which is why the seven-second preview and the full database must never share a `CHECKPOINT_DIR`. `SLICES` on `daneel.sh` shards a plan by test track, one container per slice, which is the only way to make a plan use more than one core, a cell being a single Python process; the slice is read inside the `N_TEST` selection and the training pool fixed by `N_TRAIN` and `SEED`, so shards stay comparable and their journals merge by track. `musdb` reads stems through `stempeg`, which hard-fails without `ffmpeg` on the path. `matplotlib` sits in its own `figures` dependency group rather than in `dev`, which `uv run` installs by default: a figure is drawn once on a laptop and every experiment container would otherwise download it for nothing.

## Caveats

The journals are diagnostics sized to tell one hypothesis from another, on one corpus, without confidence intervals. `report_ceiling.py` pairs every estimator against the oracles of the same track and reports the distance to the ceiling rather than a gain over the mixture, which is the whole point of the repository, but a distance to `m*` is a distance to the unconstrained real gain: the `positive` and `clipped` rows are the ones a bounded-mask system should be read against. The full-covariance arm at `nfft = 1024` is out of arithmetic reach and the plans say so rather than pretending otherwise, every published measurement being diagonal.
