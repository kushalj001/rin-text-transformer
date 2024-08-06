# Adapting RINs for text

## Updates in [commit](https://github.com/kushalj001/rin-text-transformer/commit/e5ee9e03afc56a87fb7cb783ed26380a09d28cab): 
* All the new changes have been added in the notebook (text-RIN). Only refer to that if you're going through the codebase. The py files have not been updated. I will do that once I have a running model that actually trains.
* What was added in terms of components:
  * RIN module (more complete than last time)
  * Text diffuison module (mixes [diffusion LM](https://github.com/madaan/minimal-text-diffusion) codebase for text with RIN)
  * Dataset class to load wikitext2. I was initially planning to load datasets that were used in earlier diffusion papers, but those datasets are pretty small and the tasks are different (conditional generation). My first goal is to test reconstruction capability of this model and for that I've gone with a bigger dataset.
  * Minimal training code: just a small loop that does not run yet end-to-end
  * Tokenization: I've just used gpt2 tokenizer for now to create the dataset. If we train a new tokenizer on wikitext2, we could probably reduce the vocabulary size from 50k. Will come back to this.
 
Roadmap (P0):  

- [ ] Complete a very simple working training codebase. First step is to train and overfit the model on a small dataset and then go on to do more fancy stuff with the code.  
- [ ] Figure out how to process "t" in the code. Currently it is only passed to latent side. The information at some point does get mixed with interface but probably it should be passed on both sides.  
- [ ] Add positional embeddings to interface tokens (currently only done on latent side).  
- [ ] Revisit positional embeddings in general. On latent side, there's some learned sinusoidal embeddings being used currently. See if we can use something simpler, more robust like absolute positional embeddings on both latent and interface side.  
- [ ] How latent self conditioning is being handled currently. It's taken from RIN codebase but verify if it makes sense overall.    
- [ ] Noise schedules: check which noise schedule to use. We currently have 2 sets from RIN image and text codebase. And both of these result in quite different plots. Verify which is more correct for our use-case.  

P1:  
- [ ] Distributed training code  
- [ ] Validation set, loss reporting, wandb etc.  

### Resources/links:
* https://github.com/madaan/minimal-text-diffusion
* https://github.com/justinlovelace/latent-diffusion-for-language/tree/main
* https://github.com/XiangLi1999/Diffusion-LM
* https://github.com/lucidrains/recurrent-interface-network-pytorch

  
--------
## Updates in [commit](https://github.com/kushalj001/rin-text-transformer/commit/5a424dd0ba0ce3b3a43150bc457226feb8260b43)
* Files with `annotated*` are files taken from lucidrains's image RIN codebase. These files have annotations/notes that I made while reading the code.
* `rin_text.py` is the code I am currently working on, which adapts the image codebase to text.
* The jupyter notebook is basically where I am actively working/experimenting. For now I am just appending whatever I write and test in the notebook to the file above. The .py file cannot be executed in the current state. More notes can be found in the notebook.
