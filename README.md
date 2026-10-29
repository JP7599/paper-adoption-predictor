# Paper Adoption Predictor, the app

New biotech preprints from bioRxiv, ranked by how likely they are to be among the 10 % most cited of their half-year, with the reasons.
A Fontys data-science student project by Jean Pierre van Driel. This repository holds only what the deployed app needs: the Streamlit app,
the `paper_adoption` package, the saved model and the current feed. The notebooks, the tests and the data pipeline live in the project
repository. Nothing here is a citation count: it is a reading aid.

Run locally: `pip install -r requirements.txt` then `streamlit run app/biotech_app.py`.
