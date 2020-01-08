python train_agent.py --data_dir=/data/yuming/eeg-rl-data/rt-norm/reinforcement/data-intra-s01 --output_dir=output-intra-s01 --gpus=0
python test_agent.py --data_dir=/data/yuming/eeg-rl-data/rt-norm/reinforcement/data-intra-s01 --checkpoint_dir=output-intra-s01 --output_dir=test-output-intra-s01 --gpus=0

# supervised
python data_prep.py --indir=sr128-train --outdir=.
python data_prep.py --indir=sr128-val --outdir=.

python train_supervised.py --data_dir=data-supervised --output_dir=output-supervised --gpus=0
python test_supervised.py --data_dir=/data/yuming/eeg-rl-data/rt-norm/reinforcement/data-intra-s01 --checkpoint_dir=output-supervised --output_dir=test-output-supervised --gpus=0

# the current best result is 6th ones
