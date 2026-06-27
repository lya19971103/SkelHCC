import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from clip import clip
from models import Base
from models.utils.losses import GAP_contrastive_loss,contrastive_loss, cls_acc,map_labels
import numpy as np
import importlib
from models.utils.prompt_tools import load_clip_to_cpu, _tokenizer, _get_clones, HiddenLayer2D, get_rank,SkelMaPLe2D
import torch.nn.functional as F
from collections import OrderedDict
import lorentz as L
import math
import glob
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from Text_Prompt import *

# Mask scaling 
BJ_mask_scale=21.0
BP_mask_scale=9.0

classes, num_text_aug, text_dict = text_prompt_openai_pasta_pool_4part()
text_list = text_prompt_openai_random()

splits = {'ntu': [  
                [2,3,4,8,20], # head interpolate + 7
                [4,5,6,7,8,9,10,11,21,22,23,24], #hands
                [0,1,4,8,12,16,20], # torso  interpolate + 5
                [0,12,13,14,15,16,17,18,19] # feet  interpolate + 3
            ],
          'kinetic':[
              [16,14,15,17,0],
              [1,2,3,4,5,6,7],
              [0,1,2,5,8,11],
              [8,11,9,12,10,13]
          ]}


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg['n_ctx']
        ctx_init = cfg['ctx_init']
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # Default is 1, which is compound shallow prompting
        self.compound_prompts_depth = cfg['prompt_depth']  # max=12, but will create 11 such shared prompts

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        # self.proj = nn.Linear(ctx_dim, 768)
        # self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # ncls, ntkn, 512
        # ncls, 77, 512
        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens # embedding length for class name only (i.e. without prompt)

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )
        
        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
        
        return prompts, self.compound_prompts_text #, visual_deep_prompts  
        
class SkelhccAdapter(Base):
    def __init__(self, emb_dim, model_args, cls_labels, few_shot, adapter_args, bj_names, bp_names, 
                 t_names, dataloader_type,
                 prompt_args, backbone_path = None, root = './', 
                 **kwargs):
        init_dict = locals().copy()
        init_dict.pop('self')
        super().__init__(**init_dict)
        
        self.loss_func = contrastive_loss
       
        # feature skel extractor
        backbone_name = model_args['backbone']
        backbone_camel_name = ''.join([i.capitalize() for i in backbone_name.split('_')])
        backbone_camel_name='Model'
        backbone = getattr(importlib.import_module('.backbones.'+backbone_name, package=__package__), backbone_camel_name)
       
        self.model = backbone()
        
        # load pretrained backbone
        print("load pretrained backbone")
        checkpoint = torch.load(backbone_path, map_location='cpu')
        
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] if k.startswith('module.') else k  # \u53bb\u9664'module.'
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict)
        self.local_type = model_args['local_type']
        self.proj = nn.Sequential(
                    SkelMaPLe2D(256, 512),
                    HiddenLayer2D(1, 512),
                    SkelMaPLe2D(512, emb_dim, .5, nn.Tanh()))
    
        self.clip = load_clip_to_cpu(root, 'clip')
        
        if prompt_args['use_text_prompt']:
            self.prompt_learner = MultiModalPromptLearner(prompt_args, cls_labels, self.clip)#.half()
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts

            tokenized_list = []
            for descriptions in self.bp_names:  
                sample_tokens = []
                for desc in descriptions:    
                    tokens = clip.tokenize(desc)
                    sample_tokens.append(tokens)
                sample_tokens = torch.cat(sample_tokens)
                tokenized_list.append(sample_tokens)
            self.tokenized_prompts_bp = torch.stack(tokenized_list)
            self.tokenized_prompts_bp=(self.tokenized_prompts_bp.permute(1,0,2))[0:4]
            
        else:
            classnames = [name.replace("_", " ") for name in cls_labels]
            self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in classnames])

        self.dtype = self.clip.dtype
        self.text_encoder = self.clip.encode_text

        # tmp learner
        self.tmp = nn.Parameter(
                        torch.ones([], dtype=torch.float
                    ) * np.log(1 / 0.07), requires_grad=True)
        
        self.mask_learner = nn.ModuleDict()
        self.mask_key = {}
        if 'bj' in self.local_type:
            self.mask_learner['bj'] = nn.Sequential(
                                        nn.Linear(512, 128, bias=False),
                                        nn.ReLU(),
                                        # nn.Linear(128, 128, bias=False),
                                        # nn.ReLU(),
                                        nn.Linear(128, 25, bias=False),
                                        nn.Tanh(),
                                    )
            self.mask_key['bj'] = bj_names
        if 'bp' in self.local_type:
            self.mask_learner['bp'] = nn.Sequential(
                                        nn.Linear(512, 64, bias=False),
                                        nn.ReLU(),
                                        # nn.Linear(64, 64, bias=False),
                                        # nn.ReLU(),
                                        nn.Linear(64, 4, bias=False),
                                        nn.Tanh(),
                                    )  
            self.mask_key['bp'] = bp_names

        
        # name_to_update = "prompt_learner"
        for name, param in self.named_parameters():
            # if name_to_update not in name:
            # Make sure that VPT prompts are updated
            if "clip" in name:
                param.requires_grad_(False)

        self.model.requires_grad_(False)
        self.model.eval()
        self.clip.eval()


        self.beta, self.alpha1,self.gamma1,self.lamda1 = nn.Parameter(
                        torch.ones([], dtype=torch.float
                    ) * adapter_args['beta'], requires_grad=True), nn.Parameter(
                        torch.ones([], dtype=torch.float
                    ) * adapter_args['alpha'], requires_grad=True), nn.Parameter(
                        torch.ones([], dtype=torch.float
                    ) * adapter_args['gamma'], requires_grad=True), nn.Parameter(
                        torch.ones([], dtype=torch.float
                    ) * adapter_args['lamda'], requires_grad=True)

        self.logit_scale = nn.Parameter(
                        torch.ones([], dtype=torch.float
                    ) * np.log(1 / 0.07), requires_grad=True)

        self.skeletal_alpha = nn.Parameter(torch.tensor(512 ** -0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(512 ** -0.5).log())

        self.skeletal_alpha_p = nn.Parameter(torch.tensor(512 ** -0.5).log())
        self.skeletal_alpha_q = nn.Parameter(torch.tensor(512 ** -0.5).log())

        self.curv = nn.Parameter(
            torch.tensor(0.1).log(), requires_grad=False
        )

        # When learning the curvature parameter, restrict it in this interval to
        # prevent training instability.
        self._curv_minmax = {
            "max": math.log(0.1 * 10),
            "min": math.log(0.1 / 10),
        }
           
    def configure_optimizers(self, monitor1, monitor2, lr=1e-3):
        params = self.parameters()

        optimizer = optim.Adam(params, lr=lr)

        scheduler = lrs.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=25, 
                                        cooldown=3)
        scheduler = {'scheduler': scheduler,
                    'monitor': monitor1,
                    'mode': 'min'
                    }

        return [optimizer], [scheduler]

    def pq_hdist(self, ske_p,ske_q):

        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()
        
        ske_q_V=(ske_q.mean(2)).reshape(ske_q.shape[0], -1)
        ske_p_V=(ske_p.mean(2)).reshape(ske_p.shape[0], -1)

        ske_q=ske_q.reshape(ske_q.shape[0], -1)
        ske_p=ske_p.reshape(ske_p.shape[0], -1)
    
        ske_q_emb_global = L.exp_map0(ske_q,_curv)
        ske_p_emb_global = L.exp_map0(ske_p,_curv)

        ske_q_emb_V = L.exp_map0(ske_q_V)
        ske_p_emb_V = L.exp_map0(ske_p_V)

        logits_h_pq = -L.pairwise_dist(ske_q_emb_global, ske_p_emb_global, _curv)
        logits_h_pq_V =  -L.pairwise_dist(ske_q_emb_V,ske_p_emb_V,_curv)

        return  logits_h_pq.softmax(dim=-1),logits_h_pq_V.softmax(dim=-1)

    def Text_Skeleton_CL(self, skeleton_embedding, text_embedding, text_embedding_bp, tmp, label, reduction='mean', verify_hyperbolicity=True):
        N, C, T, V = skeleton_embedding.shape

        raw_global = skeleton_embedding.view(N, C, -1).mean(dim=2) 
        
        raw_bj = skeleton_embedding.mean(dim=2).permute(0, 2, 1).reshape(-1, C)

        ske_emb_local_bp_temp = skeleton_embedding.mean(dim=2) # [N, C, V]
        bp_feats_list = []

        for i in range(len(splits['ntu'])):

            part_feat = ske_emb_local_bp_temp[:, :, splits['ntu'][i]].mean(dim=-1)
            bp_feats_list.append(part_feat)
        
        selected_bps = [bp_feats_list[0], bp_feats_list[1], bp_feats_list[3]]
        raw_bp = torch.stack(selected_bps, dim=1).reshape(-1, C)

        ske_emb_global = raw_global 
        txt_emb_global = text_embedding

        ske_emb_local_bj = skeleton_embedding.mean(-2).reshape(N, -1)
        txt_emb_local_bj = text_embedding.unsqueeze(-1).expand(*text_embedding.shape[:2], V).reshape(N, -1)
        
        ske_emb_local_bp_raw = skeleton_embedding.mean(-2) 
        add_ske_emb_local_bp = []
        for i in range(len(splits['ntu'])):
            add_ske_emb_local_bp.append(ske_emb_local_bp_raw[:, :, splits['ntu'][i]].mean(-1).unsqueeze(-1))
        
        ske_emb_local_bp = torch.cat([add_ske_emb_local_bp[0], add_ske_emb_local_bp[1], add_ske_emb_local_bp[3]], dim=-1).reshape(N, -1)
        
        mask = torch.tensor([True, True, False, True])
        txt_emb_local_bp = text_embedding_bp[:, mask, :].permute(0, 2, 1).reshape(N, -1)

        self.skeletal_alpha.data = torch.clamp(self.skeletal_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()

        # Exponential Map
        ske_emb_global = L.exp_map0(ske_emb_global * self.skeletal_alpha.exp(), _curv)
        txt_emb_global = L.exp_map0(txt_emb_global * self.textual_alpha.exp(), _curv)
        ske_emb_local_bj = L.exp_map0(ske_emb_local_bj * self.skeletal_alpha.exp(), _curv)
        txt_emb_local_bj = L.exp_map0(txt_emb_local_bj * self.textual_alpha.exp(), _curv)
        ske_emb_local_bp = L.exp_map0(ske_emb_local_bp * self.skeletal_alpha.exp(), _curv)
        txt_emb_local_bp = L.exp_map0(txt_emb_local_bp * self.textual_alpha.exp(), _curv)

        # Pairwise Distance
        logits_per_skel_txt_global = -L.pairwise_dist(ske_emb_global, txt_emb_global, _curv)
        logits_per_txt_skel_global = -L.pairwise_dist(txt_emb_global, ske_emb_global, _curv)
        logits_per_skel_txt_local_bj = -L.pairwise_dist(ske_emb_local_bj, txt_emb_local_bj, _curv)
        logits_per_txt_skel_local_bj = -L.pairwise_dist(txt_emb_local_bj, ske_emb_local_bj, _curv)
        logits_per_skel_txt_local_bp = -L.pairwise_dist(ske_emb_local_bp, txt_emb_local_bp, _curv)
        logits_per_txt_skel_local_bp = -L.pairwise_dist(txt_emb_local_bp, ske_emb_local_bp, _curv)

        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        _scale = self.logit_scale.exp()

        # Loss Calculation
        loss_global = 0.5 * (
            nn.functional.cross_entropy(_scale * logits_per_skel_txt_global, label)
            + nn.functional.cross_entropy(_scale * logits_per_txt_skel_global, label)
        )
        loss_local_bj = 0.5 * (
            nn.functional.cross_entropy(_scale * logits_per_skel_txt_local_bj, label)
            + nn.functional.cross_entropy(_scale * logits_per_txt_skel_local_bj, label)
        )
        loss_local_bp = 0.5 * (
            nn.functional.cross_entropy(_scale * logits_per_skel_txt_local_bp, label)
            + nn.functional.cross_entropy(_scale * logits_per_txt_skel_local_bp, label)
        )

        # Entailment Loss
        _angle_global = L.oxy_angle(txt_emb_global, ske_emb_global, _curv)
        _aperture_global = L.half_aperture(txt_emb_global, _curv)
        entailment_loss_global = torch.clamp(_angle_global - 0.7 * _aperture_global, min=0).mean()

        _angle_local_bj = L.oxy_angle(txt_emb_local_bj, ske_emb_local_bj, _curv)
        _aperture_local_bj = L.half_aperture(txt_emb_local_bj, _curv)
        entailment_loss_local_bj = torch.clamp(_angle_local_bj - 0.7 * _aperture_local_bj, min=0).mean()

        _angle_local_bp = L.oxy_angle(txt_emb_local_bp, ske_emb_local_bp, _curv)
        _aperture_local_bp = L.half_aperture(txt_emb_local_bp, _curv)
        entailment_loss_local_bp = torch.clamp(_angle_local_bp - 0.7 * _aperture_local_bp, min=0).mean()

        entailment_loss_local = (self.lamda1 * entailment_loss_local_bj + self.gamma1 * entailment_loss_local_bp) * 0.5
        loss_local = (self.lamda1 * loss_local_bj + self.gamma1 * loss_local_bp) * 0.5

        loss = (self.alpha1 * (loss_global + 0.2 * entailment_loss_global) + loss_local + 0.2 * entailment_loss_local) * 0.05
        logit = self.alpha1 * logits_per_skel_txt_global + self.lamda1 * logits_per_skel_txt_local_bj + self.gamma1 * logits_per_skel_txt_local_bp

        return loss, logit

    def loss(self, support_x, support_y, support_z, 
             query_x, query_y, query_z, 
             curr_epoch, is_train=True, is_test=False, **kwargs):

        ntu120_bp_mask=np.load('./data/ntu120_bp_weights.npy')
        ntu120_bp_mask=BP_mask_scale * ntu120_bp_mask 

        ntu120_bj_mask=np.load('./data/ntu120_bj_weights.npy')
        ntu120_bj_mask=BJ_mask_scale * ntu120_bj_mask 

        if  is_train: 

            self.train()
            
            support_x = support_x.view(-1, *support_x.shape[2:])
            tmp_support_feature=self.model(support_x)
            tmp_support_feature=self.proj((tmp_support_feature.permute(0,2,3,1), [], 0))[0].permute(0,3,1,2)
            support_features = tmp_support_feature.mean(-1).mean(-1)

            tmp_query_feature= self.model(query_x)
            tmp_query_feature=self.proj((tmp_query_feature.permute(0,2,3,1), [], 0))[0].permute(0,3,1,2)
            query_features = tmp_query_feature.mean(-1).mean(-1)
            
            num_support = support_features.shape[0]
            num_query = query_features.shape[0]

            proj_support_features = support_features 
            proj_query_features = query_features 

            h_logit,h_logit_V=self.pq_hdist(tmp_support_feature,tmp_query_feature)
            tmp_text_bp=(self.tokenized_prompts_bp.to(support_features.device).permute(1,2,0))[support_z]#.view(-1, *text_features.shape[1:])
            tmp_text_bp=tmp_text_bp.reshape(-1,77,4).permute(0,2,1).flatten(0,1)

            att_emb_bp= self.text_encoder(tmp_text_bp).reshape(num_support,4,512).float()
            text_features = self.text_encoder(self.tokenized_prompts.to(support_features.device)).float()
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)


            att_emb = text_features[support_z].view(-1, *text_features.shape[1:]) # shape: #att_emb: torch.Size([20, 512])
    
            batch_labels = len(support_features) * get_rank() + torch.arange( len(support_features), device= support_features.device)

            loss, support_clip_logits = self.Text_Skeleton_CL(tmp_support_feature, att_emb, att_emb_bp,
                                                        self.tmp, batch_labels)

            att_emb2 = text_features[query_z].view(-1, *text_features.shape[1:]) # shape: #att_emb2: torch.Size([20, 512])
            batch_labels2 = len(query_features) * get_rank() + torch.arange(len(query_features), device= query_features.device)

            
            loss_q, clip_logits = self.Text_Skeleton_CL(tmp_query_feature, att_emb, att_emb_bp,
                                                        self.tmp, batch_labels2)

            # Mask loss
            emphasize_labels = {}
            for key in self.mask_learner.keys():
                if key =='bj':
                    
                    emphasize_labels[key] = torch.tensor([ntu120_bj_mask[z] for z in support_z]).unsqueeze(0).to(support_features.device)#.squeeze(1)

                if key=='bp' :

                    emphasize_labels[key] = torch.tensor([ntu120_bp_mask[z] for z in support_z]).unsqueeze(0).to(support_features.device)#.squeeze(1)
                        
            # Tip adapter loss
            for key in self.mask_learner.keys():
                if key == 'bj':
                    bj_query_features0 = tmp_query_feature.mean(-2)
                    bj_support_features0 = tmp_support_feature.mean(-2)
                    curr_emphasize_predict_bj = emphasize_labels[key].squeeze(0)                  
                   
                    
                    bj_support_features = bj_support_features0.unsqueeze(0).repeat(len(bj_query_features0), 1, 1, 1) \
                                            .permute(0, 1, 3, 2).contiguous() # shape: (n*n_shot, 18, 256)

                    bj_query_features = bj_query_features0.unsqueeze(1).repeat(1, len(bj_support_features0), 1, 1) 
                    bj_query_features = bj_query_features.permute(0, 1, 3, 2).contiguous() # shape: (n, n*n_shot, 256, 18)

                    affinity_bj = (bj_query_features * bj_support_features).mean(-1) # shape: (n, n*n_shot, 25, 256)

                elif key == 'bp':
                    
                    bp_support_features = tmp_support_feature.mean(-2) # shape: (n*n_shot, 256, 25)
                    bp_query_features = tmp_query_feature.mean(-2)
                    curr_emphasize_predict_bp = emphasize_labels[key].squeeze(0)

                    add_support_features = []
                    add_query_features = []
                    for i in range(len(splits['ntu'])):
                        add_support_features.append(bp_support_features[:, :, splits['ntu'][i]].mean(-1).unsqueeze(1))
                        add_query_features.append(bp_query_features[:, :, splits['ntu'][i]].mean(-1).unsqueeze(1))
                    
                    
                    bp_support_raw = torch.cat(add_support_features, dim=1)  # [num_support, 4, C]
                    bp_query_raw = torch.cat(add_query_features, dim=1)      # [num_query, 4, C]

                    bp_support_features = bp_support_raw.unsqueeze(0).expand(
                        num_query, -1, -1, -1
                    )
                    bp_query_features = bp_query_raw.unsqueeze(1).expand(
                        -1, num_support, -1, -1
                    )
                    affinity_bp = (bp_query_features * bp_support_features).mean(-1)
                    
            affinity_bj1 = affinity_bj * curr_emphasize_predict_bj
            affinity_bj1  =  affinity_bj1.sum(-1).float() 

            affinity_bp1 = affinity_bp * curr_emphasize_predict_bp 
            affinity_bp1  =  affinity_bp1.sum(-1).float() 


           
            cache_keys = proj_support_features.view(-1, *proj_support_features.shape[1:]) 
            affinity_global = (proj_query_features @ cache_keys.T).softmax(dim=-1) 
    
            affinity = torch.cat([affinity_global, affinity_bj1, affinity_bp1],dim=1).view(num_query, 3, num_support).transpose(1, 2).flatten(1, 2)
            cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp()
            weights=torch.stack([self.alpha1,self.lamda1,self.gamma1])*0.1
            
            cache_values = F.one_hot(support_y.expand(-1,1).repeat(1,3).view(-1), num_classes=len(support_z)).float()
            cache_values = cache_values.view(-1, 3, len(support_z))  
            cache_values = cache_values * weights.view(1, 3, 1)  
            cache_values = cache_values.view(-1, len(support_z))  

            cache_logits = cache_logits @ cache_values

            tip_logits = 6.0*clip_logits + cache_logits 

            loss = loss + F.cross_entropy(tip_logits, query_y)

            acc, pred = cls_acc(tip_logits, query_y)
              
        else:
      
            self.eval()
          	
            true_class_ids = list(dict.fromkeys(list(x for x in support_z)))
            support_y = torch.tensor([true_class_ids.index(x) for x in support_z]).to(support_x.device)

            true_class_ids_q = list(dict.fromkeys(list(x for x in query_z)))
            query_y = torch.tensor([list(support_z).index(x) for x in true_class_ids_q]).to(support_x.device)

            tmp_support_feature=self.model(support_x)
            tmp_support_feature=self.proj((tmp_support_feature.permute(0,2,3,1), [], 0))[0].permute(0,3,1,2)
            support_features = tmp_support_feature.mean(-1).mean(-1)

            tmp_query_feature= self.model(query_x)
            tmp_query_feature=self.proj((tmp_query_feature.permute(0,2,3,1), [], 0))[0].permute(0,3,1,2)
            query_features = tmp_query_feature.mean(-1).mean(-1)


            num_support = support_features.shape[0]
            num_query = query_features.shape[0]
            
            proj_support_features = support_features 
            proj_query_features = query_features 
            
            proj_support_features = F.normalize(proj_support_features, dim=-1)
            proj_query_features = F.normalize(proj_query_features, dim=-1)
            
            text_features = self.text_encoder(self.tokenized_prompts.to(proj_support_features.device)).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            support_att_emb = text_features[support_z].view(-1, *text_features.shape[1:]) 

            tmp_text_bp=(self.tokenized_prompts_bp.to(support_features.device).permute(1,2,0))[support_z]#.view(-1, *text_features.shape[1:])
            tmp_text_bp=tmp_text_bp.reshape(-1,77,4).permute(0,2,1).flatten(0,1)
            support_att_emb_bp= self.text_encoder(tmp_text_bp).reshape(num_support,4,512).float()
            support_att_emb_bp = support_att_emb_bp / support_att_emb_bp.norm(dim=-1, keepdim=True)

            batch_labels = len(proj_support_features) * get_rank() + torch.arange(len(proj_support_features), device= proj_support_features.device)

            loss, clip_logits = self.Text_Skeleton_CL(tmp_query_feature, support_att_emb,support_att_emb_bp,self.tmp, query_y)
            
            # Mask loss
            emphasize_labels = {}
            for key in self.mask_learner.keys():
                if key =='bj':
                    
                    emphasize_labels[key] = torch.tensor([ntu120_bj_mask[z] for z in support_z]).unsqueeze(0).to(support_features.device)

                if key=='bp' :

                    emphasize_labels[key] = torch.tensor([ntu120_bp_mask[z] for z in support_z]).unsqueeze(0).to(support_features.device)

            # Tip adapter loss
            for key in self.mask_learner.keys():
                if key == 'bj':
                    bj_query_features0 = tmp_query_feature.mean(-2)
                    bj_support_features0 = tmp_support_feature.mean(-2)
                    curr_emphasize_predict_bj = emphasize_labels[key].squeeze(0)
                  
                    bj_support_features = bj_support_features0.unsqueeze(0).repeat(len(bj_query_features0), 1, 1, 1) \
                                            .permute(0, 1, 3, 2).contiguous() # shape: (n*n_shot, 18, 256)
                    bj_query_features = bj_query_features0.unsqueeze(1).repeat(1, len(bj_support_features0), 1, 1) 
                    bj_query_features = bj_query_features.permute(0, 1, 3, 2).contiguous() # shape: (n, n*n_shot, 256, 18)
                    affinity_bj = (bj_query_features * bj_support_features).mean(-1) # shape: (n, n*n_shot, 25, 256)
                    
                elif key == 'bp':
                    
                    bp_support_features = tmp_support_feature.mean(-2) # shape: (n*n_shot, 256, 25)
                    bp_query_features = tmp_query_feature.mean(-2)
                    curr_emphasize_predict_bp = emphasize_labels[key].squeeze(0)

                    add_support_features = []
                    add_query_features = []
                    for i in range(len(splits['ntu'])):
                        add_support_features.append(bp_support_features[:, :, splits['ntu'][i]].mean(-1).unsqueeze(1))
                        add_query_features.append(bp_query_features[:, :, splits['ntu'][i]].mean(-1).unsqueeze(1))
                    
                    bp_support_features = torch.cat(add_support_features, dim=1).unsqueeze(0).repeat(len(bp_query_features), 1, 1, 1) 
                    bp_query_features = torch.cat(add_query_features, dim=1).unsqueeze(1).repeat(1, len(bp_support_features), 1, 1)                      
                    affinity_bp = (bp_query_features * bp_support_features).mean(-1)


            affinity_bj1 = affinity_bj * curr_emphasize_predict_bj 
            affinity_bj1  = affinity_bj1.sum(-1).float()          

            affinity_bp1 = affinity_bp * curr_emphasize_predict_bp 
            affinity_bp1 = affinity_bp1.sum(-1).float()

            cache_keys = proj_support_features.view(-1, *proj_support_features.shape[1:]) 
            affinity_global = (proj_query_features @ cache_keys.T).softmax(dim=-1) 
           
            affinity = torch.cat([affinity_global, affinity_bj1, affinity_bp1],dim=1).view(num_query, 3, num_support).transpose(1, 2).flatten(1, 2)
            cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp()
            
            cache_values=F.one_hot(support_y.unsqueeze(1).expand(-1,1).repeat(1,3).view(-1), num_classes=len(support_z)).float()
            weights=torch.stack([self.alpha1,self.lamda1,self.gamma1],dim=0)*0.1
           
            cache_values = cache_values.view(-1, 3, len(support_z))  
            cache_values = cache_values * weights.view(1, 3, 1)  
            cache_values = cache_values.view(-1, len(support_z))  

            cache_logits=cache_logits @ cache_values
           
            tip_logits =  6.0*clip_logits  + cache_logits
           
            acc, pred = cls_acc(tip_logits, query_y)
            loss =  F.cross_entropy(tip_logits, query_y)
            
        return {'loss': loss}, {'acc': acc, 'preds': pred, 'alpha': self.alpha1.cpu().item(), 'beta': self.lamda1.cpu().item(),'gamma': self.gamma1.cpu().item()}
