"""
Command-line tools, run as `python -m robomimic.scripts.<name>`:

- `make_datasets`: robomimic low_dim datasets of the transferred demonstrations.
- `download_datasets`: the released robomimic datasets, from Hugging Face.
- `transfer_demos`: transfer a dataset's demonstrations into the MuJoCo Warp POMDP.
- `add_observations`: add the POMDP's observations to a dataset of recorded states.
- `replay_demos`: videos of demonstrations replayed open-loop.
- `get_dataset_info`, `split_train_val`, `filter_dataset_size`: inspect and split robomimic datasets.
"""
