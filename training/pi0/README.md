# π₀ training variant

This experiment adds language-conditioned FiLM to image tokens and training-only state-token dropout/noise. It was developed against LeRobot commit `ae2b57f12223c8c5184fcd814eb63cd9ce7de203`. Apply [the Pi0 patch](../patches/lerobot_pi0_film_state_noise.patch) to that LeRobot revision, install LeRobot in editable mode, and supply an existing base checkpoint through `PI0_BASE_CHECKPOINT` before running `train_pi0_film.sh`. Weights are stored outside this repository.

`--steps=20000`, batch size 1 and gradient accumulation 4 correspond to approximately 5,000 optimizer updates. The script starts a new run; it is not an optimizer-state resume command. State dropout/noise are training-only and do not establish safe real-robot behaviour.
