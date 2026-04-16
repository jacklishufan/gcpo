Design docs for guidance gap.


Goal, implement per-token weighting in RL loss by calculating the per token importance as f(logp(y|x+),logp(y|x-)), where x+, y is the prompt and generation pairs and x- is some "negative prompt". 

I want to define three variants in a way that is easily expandable. The cureent inclusions should be 
1. x- is the same prompt x plus the text ", but generate the wrong answer"
2. x- is the same prompt with an empty image white
3. x- is an empty prompt with no content (i.e., just "answer the question")

You should refer to related implementation in /home/schmidt/ssci-shufan/VPPO-RL, which implement VPPO, where a specific choice of f is used and x- is the same prompt x with augmented image, try to follow the general logic but do note that what we are trying to build is different.

Some guides:

verl/workers/fsdp_workers.py implements loglikelyhood query, and is also where data aug should happen

erl/workers/actor/dp_actor.py is where we acuatly compute and apply per token weighting

for now, implement three f: kl, information_gain and abs

specifica defs and normalizations can be find in demo/visualize_importance_weights_ar.py

Some important things to note:
makse sure output logps are aligned, since input prompt may differ in length