# Irish_Eng_Training
Prior to GUI repo, this is the scripts used to rate  translations of the parallel Irish and English data with a reference free model, and format it into a preference dataset.


Data downloaded from Tatoeba (community generated pairs) and Gaois (legislation pairs).

```
585265,Go raibh míle maith agat!,
1564,Thank you very much!
```

The two datasets are in differing formats, so need two separate notebooks to format them (for example, Gaois is very large after concatenating the `thx` files, so we take a subsample).

1. `irish_eng_data_final_1st_ds_Gaois.ipynb
2. `irish_eng_data_2nd_ds_Tatoeba.ipynb

We then use the GPT4o API to translate in both directions and saved as `translated.jsonl`

```
{"en": "DAIRY PRODUCE (PRICE STABILISATION) ACT, 1933", 
"ga": "ACHT TORA DÉIRÍOCHTA (PRAGHAS DO DHÉANAMH SEASMHACH), 1933.", 
"gpt_4_ga": "ACHT UM SHOCRÚ PRAGHAS TÁIRGÍ DAIRÍ, 1933", 
"gpt_4_en": "DESTRUCTION OF WEEDS (STABILIZATION OF PRICE) ACT, 1933."}
```

After dataset is stored locally, run the `download_cometxxl.py` which downloads the reference comet model free to evaluate translations. The model from provided Huggingface Hub path, in this case `Unbabel/wmt23-cometkiwi-da-xxl`.

Once data model is downloaded to a Modal Labs Volume like so:

```
┏━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Filename       ┃ Type ┃ Created/Modified                   ┃ Size     ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
┡━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ checkpoints    │ dir  │ 2024-12-30 17:14 GMT Standard Time │ 0 B      │
│ README.md      │ file │ 2024-12-30 17:14 GMT Standard Time │ 4.0 KiB  │
│ LICENSE        │ file │ 2024-12-30 17:14 GMT Standard Time │ 20.3 KiB │
│ .gitattributes │ file │ 2024-12-30 17:14 GMT Standard Time │ 1.5 KiB  │
│ .cache         │ dir  │ 2024-12-30 17:14 GMT Standard Time │ 11 B     │
└────────────────┴──────┴────────────────────────────────────┴──────────┘
```

We grade the translations `script_cometxl_scorer.py `which also formats the dataset into what cometxl expects (with `src` and `mt` columns). The `system_score` is there to average multiple reference model scores, but in this case we only used one model). The reference free model then grades the translations between 0-100 with closer to 100 being a 'perfect' translation.

```
{"src": "Thank you very much!", "mt": "Go raibh míle maith agat!", 
"direction": "en-ga", 
"cometkiwi_score": 0.8342427015304565, 
"system_score": 0.8342427015304565}

{"src": "Go raibh míle maith agat!", 
"mt": "Thank you very much!", "direction": "ga-en", "cometkiwi_score": 0.8411319851875305, "system_score": 0.8411319851875305}
```

Once graded, we format it a final time to what DPO would expect using `script_preference_ds_formatter.py` and the higher rated translations are placed in `accepted` and the latter in `rejected`. 

```
{"prompt": "Search Cuardach", 
"chosen": "Cuardach", 
"rejected": "Cuardach Search"}

{"prompt": "Cuardach Search", 
"chosen": "Search Cuardach", 
"rejected": "Search"}
```

The dataset is ready to finetune a chosen model via Axolotl (hosted on Jarvis Labs).
