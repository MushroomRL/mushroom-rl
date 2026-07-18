# Neural maximization-bias result provenance

The four arrays in `source_arrays` were copied without modification from:

- Repository: https://github.com/shreyassr123/Double-SOR-Q-Learning
- Commit: `8a26c4ca37f2447b74d10572e57d5e4ef16a57a4`
- Directory: `Deep RL Version/Maximization Bias`
- Original experiment: `nn_biasexample.py`

Each array has shape `(400, 1000)`: 400 training episodes by 1,000
independent iterations. The benchmark script checks this shape before using
the data.

SHA-256 checksums:

- `ProbLeft-Q`: `7B7E5025684C3B821DEBE3F563BD764D10484AC92C8C5A72376BE78FFADFDB87`
- `ProbLeft-SORQ`: `09B183093B68C7868FBC8095C725539AE5607F3E4FA65899FD28E313B26B80B4`
- `ProbLeft-D-Q-average`: `A87D30964A0165260AE409F37C74D6A0F0B5BFE180C978AF6FD177954636F34D`
- `ProbLeft-SORDQ-average`: `B29D9353FDF0A17A1787E5884FA824E0A807AC554C4E99856C8C765B1DBDBF91`
