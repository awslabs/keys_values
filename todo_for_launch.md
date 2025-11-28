# Missing Steps for Open Source Launch

Here, we collect action items for what remains to be done for the initial
open source launch. This should be done with minimal effort.

## Mandatory Org

- [ ] Security review:<br>
  https://talos.security.aws.a2z.com/#/talos/engagement/arn:aws:talos-engagement:engagement/67eb596f-f747-47b6-be5b-1057466e46ea <br>
  https://t.corp.amazon.com/V1939587837
- [ ] Look at this and run:<br>
  https://w.amazon.com/bin/view/AWS/Patching/GitHub_Project_Security/


## Code Changes

- [x] Gradient computation with padded-query SDPA. Branch `grad_new`
- [x] Submit PR to `litgpt` with all remaining changes. Tell them why!
- [x] Fix test: `kvcache/test_quantization.py -- test_quantization_error`:<br>
  Fails on many of the GPU cases!
- [x] Remove complexity w.r.t. `device`, `dtype` args
- [x] Performance improvements
- [x] Remove `input_pos` argument
- [x] CPU offloading. This is not essential
- [ ] Identify unmatched pack args in `test_gradient.py` with new
  training replay cache variant. This is not essential.
- [ ] Fix bug with `qh2o-bnb-quantized4`: Does this still happen? Cover with test!
- [ ] Collect constants -> `keys_values/constants.py`


## GitHub

- [ ] Browse `Syne Tune` setup: What do we want here as well?
  Write down list of steps

- pyproject.toml [OK]
  Also: keys_values/version, keys_values/__init__.py: read_version
- setup.cfg [OK]
- githooks/pre-commit [OK]
- .github/*
- .github/workflows/*

Ignore for now:

- release.sh


## Clean-Up

- [x] Add missing comments for most important classes
- [ ] Remove DEBUG and TODO
- [ ] Scan for old naming / internal comments
- [ ] Ensure that licence header is everywhere


## Documentation

- [x] Installation must work with `LitGPT` `main` branch. Maybe even with `pip`?
- [x] Extend `README.md` (copy elements from `Syne Tune`)
- [ ] Write `CONTRIBUTING.md` (copy elements from `Syne Tune`)
