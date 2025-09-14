## install

cd /llm_oracle

pip install -e .

## use in python

import oracle

ogent = oracle.Oracle("gpt-4o")

ogent.query("agent", "hi")
