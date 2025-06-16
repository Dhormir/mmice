# T5
# First stage
python test_1.py -task imdb -stage1_exp mice-test-gradient-optimization -mask_type grad
# Second Stage
python test_2.py -task imdb -editor_path results/imdb/editors/mice-test-editor-length-max/checkpoints -stage2_exp mice-test-editor-max-length -n_samples 500
# MT5
# First Stage
python test_1.py -task imdb -stage1_exp mmice-test-editor-length-max -mask_type grad -model_name google/mt5-small -train_batch_size 6 -val_batch_size 6 -lr 1e-3
# Second Stage
python test_2.py -task imdb -editor_path results/imdb/editors/mmice-test-editor-length-max/checkpoints -stage2_exp mmice-test-editor-length-max -n_samples 500 -model_name google/mt5-small
# Bert
# First Stage
python test_1.py -task imdb -stage1_exp mmice-test-editor-bert -mask_type grad -model_name bert-base-uncased -train_batch_size 6 -val_batch_size 6 -lr 1e-3
# Second Stage
python test_2.py -task imdb -editor_path results/imdb/editors/mmice-test-editor-bert/checkpoints -stage2_exp mmice-test-editor-bert -n_samples 500 -model_name bert-base-uncased



python test_2.py -task imdb -editor_path results/imdb/editors/mmice-umt5-small-lora-01/checkpoints -stage2_exp mmice-umt5-small-lora-02-mauve -n_samples 10 -lora True -model_name google/umt5-small -min_metric mauve