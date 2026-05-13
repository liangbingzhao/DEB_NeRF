srun -n 1 --mem=80G  -t 10:00:00 --gres=gpu:a100:1  --pty /bin/bash
# shard02 20.34
# shard03 19.79
# shard04 28.03
# shard05 39.46
# shard06 36.1
# shard07 35.87
# shard08 39.51
# result = exa.search_and_contents(
#         "Long-Run Aggregate Supply Curves",
#         type="neural",
#         use_autoprompt=True,
#         num_results=5,
#         text=True
#         )
# result = exa.search_and_contents(
#         "hottest AI startups",
#         type="neural",
#         use_autoprompt=True,
#         num_results=10,
#         # text=True,
#         summary=True
#         )
# 195.0 / 1169  (16.7)(unseen entity)