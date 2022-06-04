# oputest

## run cuda_sample directly
app is at cuda_samples

### run smoke test firstly
- vectorCopy
- vectorAdd
- vectorSmem
- cudaTensorCoreGemm

## run pytest auto-generated test

`
cd python
pytest test/OpuIsa_test.py
or pytest test/OpuIsa_fulltest.py
`
