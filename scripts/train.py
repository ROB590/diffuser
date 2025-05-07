import diffuser.utils as utils
import pdb

# wrapper
class StateOnlyDataset:
    def __init__(self, ds):
        self.ds             = ds
        self.observation_dim = ds.observation_dim
        print("state only dataset dim is",self.observation_dim)
        self.horizon         = ds.horizon
        self.action_dim = ds.action_dim
        print("state only dataset action dim is",self.action_dim)
        self.normalizer = ds.normalizer
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # 1) get the original namedtuple sample
        sample = self.ds[idx]
        #    sample._fields might be ('observations','actions',...) or ('x','cond'), etc.

        # 2) unpack the full transition tensor and the conditioning
        #    here we assume the first element is the transition tensor:
        full_trans, cond = sample

        # 3) slice off only the state dimensions
        x_states = full_trans[:, :self.observation_dim]

        # 4) rebuild the **same** namedtuple type, replacing the tensor
        SampleType = type(sample)              # this is the namedtuple class
        new_sample = SampleType(x_states, cond)

        return new_sample
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

args = Parser().parse_args('diffusion')


#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dataset = dataset_config()
dataset = StateOnlyDataset(dataset)
############ FIX DIM
obs_norm = dataset.normalizer.normalizers['observations']
# keep only the first observation_dim entries

obs_norm.mins = obs_norm.mins[:dataset.observation_dim]
obs_norm.maxs = obs_norm.maxs[:dataset.observation_dim]
next_norm = dataset.normalizer.normalizers.get('next_observations', None)
if next_norm:
    next_norm.mins = next_norm.mins[:dataset.observation_dim]
    next_norm.maxs = next_norm.maxs[:dataset.observation_dim]
renderer = render_config()

observation_dim = dataset.observation_dim
#action_dim = dataset.action_dim #NOTE we remove the action here


#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    # transition_dim=observation_dim + action_dim,
    transition_dim=observation_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=0, # remove action so 0 
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, renderer)


#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model)

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
print('init batch),',batch)
loss, _ = diffusion.loss(*batch)
print('first loss is',loss)
loss.backward()
print('✓')


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)

