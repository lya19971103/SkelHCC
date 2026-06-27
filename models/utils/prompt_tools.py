from clip import clip
import torch
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch.nn as nn
import copy 
import torch.distributed as dist

class HiddenLayer(nn.Module):
    def __init__(self, n_hidden_layers, hidden_size):
        super().__init__()
        if n_hidden_layers != 0:
            self.hidden_layers = []
            for i in range(n_hidden_layers):
                self.hidden_layers += [SkelMaPLe(hidden_size,hidden_size)]
            self.hidden_layers = nn.Sequential(*self.hidden_layers)
        else:
            self.hidden_layers = nn.Identity()

    def forward(self, x):
        return self.hidden_layers(x)

class HiddenLayer2D(nn.Module):
    def __init__(self, n_hidden_layers, hidden_size):
        super().__init__()
        if n_hidden_layers != 0:
            self.hidden_layers = []
            for i in range(n_hidden_layers):
                self.hidden_layers += [SkelMaPLe2D(hidden_size,hidden_size)]
            self.hidden_layers = nn.Sequential(*self.hidden_layers)
        else:
            self.hidden_layers = nn.Identity()

    def forward(self, x):
        return self.hidden_layers(x)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()
   
class SkelMaPLe(nn.Module):
    def __init__(self, input_size, output_size, 
                 dropout=.5, activation=nn.SiLU()):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_size)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_size, output_size)
        self.activation = activation
        
    def forward(self, inputs):
        # analyze input
        x, compound_prompts, counter = inputs
        # preprocess
        # print(x.shape)
        x = self.dropout(self.norm(x))
        # maple
        if not (counter > len(compound_prompts) - 1):
            vis_context = compound_prompts[counter]
            vis_context = vis_context.expand(x.shape[0], -1)
            x = x + vis_context
            counter += 1
        else:
            x = x
        # linear
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        return x, compound_prompts, counter

class SkelMaPLe2D(nn.Module):
    def __init__(self, input_size, output_size, 
                 dropout=.5, activation=nn.SiLU()):
        super().__init__()
        self.norm = nn.BatchNorm2d(input_size)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_size, output_size)
        self.activation = activation
        self.input_size=input_size
    def forward(self, inputs):
        # analyze input
        x, compound_prompts, counter = inputs
        # preprocess
        # print(x.shape)
        x = self.dropout(self.norm(x.permute(0,3,1,2)))

        x = x.permute(0,2,3,1)

        # maple
        if not (counter > len(compound_prompts) - 1):
            vis_context = compound_prompts[counter]
            vis_context = vis_context.expand(x.shape[0], -1)
            x = x + vis_context
            counter += 1
        else:
            x = x
        # linear
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        return x, compound_prompts, counter

def load_clip_to_cpu(root, model_name, use_text_level_prompt=False, re_attention=False, n_ctx = 2, prompt_type=2):
    backbone_name = "ViT-B/16"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    if use_text_level_prompt:
        if model_name == 'dual_clip':
            design_details = {"trainer": 'Dual',
                            "vision_depth": 0,
                            "language_depth": 0, "vision_ctx": 0,
                            "language_ctx": 0,
                            "maple_length": n_ctx,
                            "re_attention": re_attention,
                            "prompt_type":prompt_type}
            print("Design Details:")
            print(design_details)
        else:
            design_details = {"trainer": 'MaPLe',
                        "vision_depth": 0,
                        "language_depth": 0, "vision_ctx": 0,
                        "language_ctx": 0,
                        "maple_length": n_ctx}
    else:
        design_details = None
    model = clip.build_model(state_dict or model.state_dict(), 
                             design_details)

    return model

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

_tokenizer = _Tokenizer()