# T5
# First stage
python test_1.py -task imdb -stage1_exp mice-test-gradient-optimization -mask_type grad
# Second Stage
python test_2.py -task imdb -editor_path results/imdb/editors/mice-test-editor-length-max/checkpoints -stage2_exp mice-test-editor-max-length -n_samples 500
# MT5
# First Stage
python test_1.py -task imdb -stage1_exp mmice-test-gradient-optimization -mask_type grad -model_name google/mt5-small -train_batch_size 1
# Second Stage
python test_2.py -task imdb -editor_path results/imdb/editors/mmice-test-editor-length-max/checkpoints -stage2_exp mmice-test-editor-max-length -n_samples 500 -model_name google/mt5-small