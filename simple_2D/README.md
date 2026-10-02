# simple_2D

Procedurally generated 2D grid world used earlier to develop the
observation/reward design. **Main training is no longer this toy pool.**
`train_recurrent_ppo.py` and `evaluate_policy.py` forward to the project-root
ADWA trainer: 13 real buildings for training, 4 held out for test
(`Eastville`, `Mosquito`, `Sisters2`, `Scioto2`).

```bash
cd simple_2D
# same as running ../train_adwa_ppo.py from the project root
python train_recurrent_ppo.py --timesteps 1000000
python evaluate_policy.py
```

Checkpoints land in `../checkpoints/adwa_gru_ppo.pt`.
