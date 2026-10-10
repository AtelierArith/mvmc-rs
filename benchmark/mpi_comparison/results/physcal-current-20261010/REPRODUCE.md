Source commits and output validation: see validation.json. Full inputs, outputs and logs: bench-out/physcal-current-20261010/.

```bash
bash bench/run.sh --sites 32 64 --ranks 4 --threads 4 --steps 300 --groups 100 --samples 300 --warmups 1 --reps 3 --julia-source /home/vscode/.cache/mvmc/julia-issue496 --output bench-out/physcal-reproduce
```

The Julia checkout must be at the recorded commit. Omitting --julia-source uses the repository submodule instead.
