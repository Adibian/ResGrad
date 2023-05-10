
###############################################
#################### Data #####################
batch_size = 32
target_data_dir = "dataset/Persian/resgrad_data/mel_target"
input_data_dir = "dataset/Persian/resgrad_data/mel_prediction"
durations_dir = "dataset/Persian/resgrad_data/durations"
val_size = 16
preprocessed_path = "processed_data"
normalized_method = "min-max"

shuffle_data = True
normallize_spectrum = True
min_spec_value = -13
max_spec_value = 3
normallize_residual = True
min_residual_value = -0.25
max_residual_value = 0.25
max_win_length = 100  ## maximum size of window in spectrum

###############################################
################## Training ###################
lr = 1e-4
epochs = 70
save_model_path = "output/persian/resgrad/ckpt"
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"
validate_every_n_step = 20
log_dir = 'output/persian/resgrad/log'
save_path = 'checkpoint'

###############################################
############ Model Parameters #################
model_type1 = "spec2residual"  ## "spec2spec" or "spec2residual"
model_type2 = "segment-based"  ## "segment-based" or "sentence-based"
n_feats=80
dim=64
n_spks=1
spk_emb_dim=64
beta_min=0.05
beta_max=20.0
pe_scale=1000