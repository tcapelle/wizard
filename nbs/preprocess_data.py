import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import wandb
from tqdm.auto import tqdm
from utils import parse_arg

RAW_AT = 'parambharat/wandb_docs_bot_dev/wandbot_vectorindex:latest'
OUT_FILE = Path('data/dataset.jsonl')
ENTITY = "prompt-eng"

OUT_FILE.parent.mkdir(exist_ok=True, parents=True)

config = SimpleNamespace(
    raw_at=RAW_AT,
    out_file=OUT_FILE,
    upload_table=False,
)

# parse args using custom function
config = parse_arg(config)

run = wandb.init(entity=ENTITY, project="wizard", job_type="preprocessing", config=config)

# get config and make it accessible
config = wandb.config

# download artifact without creating a run
api = wandb.Api()
artifact = api.artifact(config.raw_at, type='vectorindex')
artifact_dir = artifact.download()
path = Path(artifact_dir)

# load json
with open(path/'datastore.json', mode="r") as f:
    docstore = json.load(f)

# get docs
docs = docstore["docs"].values()

# group by source
d = defaultdict(list)
for doc in tqdm(docs, total=len(docs)):
    d[doc["extra_info"]["source"]].append(doc["text"])

# concat all docs from same source
for k in d.keys():
    d[k] = "\n".join(d[k])

# convert to list of dicts
ds_as_list = [{"source":tup[0], "text":tup[1]} for tup in d.items()]

if config.upload_table:
    N = 10
    # create wandb table
    table = wandb.Table(data=list(d.items())[0:N], columns=["source", "text"])
    
    # log table
    wandb.log({"dataset_table": table})

# dump to jsonl
with open(OUT_FILE, 'w') as outfile:
    for row in ds_as_list:
        json.dump(row, outfile)
        outfile.write('\n')

# link artifact
wandb.use_artifact(RAW_AT)

# log at
at = wandb.Artifact("lm_dataset", type="dataset")
at.add_file(OUT_FILE)
wandb.log_artifact(at)