



# TODO: sampling code (can be done later)
# double check training code is correct (check what's going on at each step)
# try to align this with text diffusion codebases
# for text, we might need an additional rounding layer with softmax before we can compare the prediction with x 
# as done currently.
class GaussianDiffusion(nn.Module):
    def __init__(
        self, 
        model, 
        timesteps,
        noise_schedule,
        latent_conditioning_prob
    ):
        super().__init__()
        self.model = model
        if noise_schedule == "linear":
            self.gamma_schedule = simple_linear_schedule
        elif noise_schedule == "cosine":
            self.gamma_schedule = cosine_schedule
        elif noise_schedule == "sigmoid":
            self.gamma_schedule = sigmoid_schedule

        self.timesteps = timesteps
        self.latent_conditioning_prob = latent_conditioning_prob

    def _right_pad_dims(self, x, t):
        padding_dims = x.ndim - t.ndim
        if padding_dims <= 0:
            return t
        return t.view(*t.shape, *((1,) * padding_dims))

    def _gamma_to_alpha_sigma(self, gamma, scale = 1):
        return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)
        
    def forward(self, x):
        # x/input_tokens = [bs, seq_len] (mostly*)
        batch, seq_len = x.shape[0], x.shape[1]
        times = torch.zeros((batch, ), device=x.device).float().uniform_(0,1)
        # time values between 0 and 1 sampled uniformly at random
        noise = torch.randn_like(x)
        # gaussian noise
        gamma = self.gamma_schedule(times)
        # get the noise values at the sampled times
        # the schedule has noise values on y axis and t on x axis.
        gamma = self._right_pad_dims(gamma, times) # What does this do?
        alpha, sigma = self._gamma_to_alpha_sigma(gamma)
        # alpha = mean coefficient of gaussian transition in a diffusion model
        # sigma = std deviation coefficient
        # TODO: write the equation here for reference, to make it clearer what this refers to
        noised_x = alpha * x_emb + sigma * noise 
        # TODO: alpha/sigma is not cumulative product of all alphas (acc to the formula)
        zt_ = None # latent estimates

        if random() < self.latent_conditioning_prob:
            with torch.no_grad():
                x0, zt_ = self.model(noised_x, times)
                zt_ = zt_.detach() # stop gradient on latent estimate
        pred = self.model(noised_x, times, zt_)
        target = x
        loss = F.mse_loss(pred, target, reduction=None)
        # TODO: check what reduction does
        # understand how this works
        loss = reduce(loss, 'b ... -> b', 'mean')
        # skipped the snr loss weights
        return loss.mean() 
    


def process_roc_dataset(dataset):
    def extract_roc_text(example):
        text = example['text']
        assert text[:2] == '["'
        assert text[-2:] == '"]'
        sentences = text[2:-2]
        return {'text': sentences}
    dataset = dataset.map(extract_roc_text, )
    dataset = dataset.shuffle(seed=42)
    # Hold out some validation samples for testing
    val_test_ds = dataset['valid'].train_test_split(train_size=1000, shuffle=False)
    dataset['valid'] = val_test_ds['train']
    dataset['test'] = val_test_ds['test']
    return dataset

roc_data_path = '../Diffusion-LM/datasets/ROCstory/'
dataset = load_dataset("text", data_files={f'{split}': os.path.join(roc_data_path, f'roc_{split}.json') for split in ['train', 'valid']})
dataset = process_roc_dataset(dataset)