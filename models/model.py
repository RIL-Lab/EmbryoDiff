import copy
import random
import time as Time

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.condition_encoder import EncoderModel
from models.diffusion_denoiser import MixedConditionDecoderV2
from models.diffusion_utils import *


class EmbryoDiff(nn.Module):
    def __init__(self, encoder_params, decoder_params, diffusion_params, num_classes, device):
        super(EmbryoDiff, self).__init__()

        self.device = device
        self.num_classes = num_classes
        self.diffusion_params = diffusion_params
        self.decoder_params = decoder_params
        self.detach_decoder = diffusion_params['detach_decoder']
        self.cond_types = diffusion_params['cond_types']

        self.configure_diffusion_params(diffusion_params)

        self.f_stage_embed = nn.Embedding(num_classes, decoder_params["num_f_maps"])

        self.sem_encoder = EncoderModel(**encoder_params)
        bound_encoder_params = copy.deepcopy(encoder_params)
        bound_encoder_params["num_classes"] = 2
        self.bound_encoder = EncoderModel(**bound_encoder_params)
        self.decoder = MixedConditionDecoderV2(**decoder_params)

    def configure_diffusion_params(self, diffusion_params):
        timesteps = diffusion_params['timesteps']
        betas = cosine_beta_schedule(timesteps)  # torch.Size([1000])
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = diffusion_params['sampling_timesteps']
        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = diffusion_params['ddim_sampling_eta']
        self.scale = diffusion_params['snr_scale']

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_sample(self, x_start, t, noise=None): # forward diffusion
        """
        add noise to the input.
        x_start: the clean input  B T C
        t: the sampled timestep
        noise: optional noise, if none, then add gaussion noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    

    def model_predictions(self, backbone_feats, x, t):
        # backbone_feats: dual encoder output dict
        # x: one-hot ground truth label during training stage
        # t: current time step

        clip_value = self.diffusion_params.get("value_clipping", 0)
        if clip_value != 0:
            x_m = torch.clamp(x, min=-1 * clip_value, max=clip_value) # [-scale, +scale]
            x_m = denormalize(x_m, self.scale)                        # [0, 1]
        else:
            x_m = x

        x_start = self.decoder(backbone_feats, t, x_m.float()) # torch.Size([1, T, C])
        x_pred = F.softmax(x_start, 2) # B T num_classes
        x_start = torch.argmax(x_pred, dim=-1) # B T
        x_start = torch.sigmoid(self.f_stage_embed(x_start))

        x_start = normalize(x_start, self.scale)                              # [-scale, +scale]
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start, x_pred
        

    def prepare_targets(self, event_gt):
        # add noise to the ground truth sequence
        # event_gt: on hot ttensor with shape (B T C) 
        # event_gt: normalized [0, 1]

        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(size=event_gt.shape, device=self.device)
        x_start = (torch.sigmoid(event_gt) * 2. - 1.) * self.scale  #[-scale, +scale]
        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        clip_value = self.diffusion_params.get("value_clipping", 0)
        if clip_value != 0:
            x = torch.clamp(x, min=-1 * clip_value, max=clip_value)
            event_diffused = ((x / clip_value) + 1) / 2.           # normalized [0, 1]
        else:
            event_diffused = x

        return event_diffused, noise, t
    

    def forward_condition_model(self, video_feats, get_features=True):
        assert len(video_feats.shape) in (3, 4), f"invalid video_feats shape {video_feats.shape}."

        if len(video_feats.shape) == 4:
            _, Focal, _, _ = video_feats.shape
            if Focal == 1:
                video_feats = video_feats.squeeze(1)
            else:
                video_feats = torch.mean(video_feats, dim=1, keepdim=False)

        sem_encoder_out, sem_encoder_feats = self.sem_encoder(video_feats, get_features=get_features)
        bound_encoder_out, bound_encoder_feats = self.bound_encoder(video_feats, get_features=get_features)

        out_dict = {
            "sem_encoder_out": sem_encoder_out,
            "sem_encoder_feats": sem_encoder_feats,
            "bound_encoder_out": bound_encoder_out,
            "bound_encoder_feats": bound_encoder_feats
        }

        return out_dict
    
    def forward(self, encoder_outs, t, event_diffused):
        """Forward pass of the diffusion decoder (training only).

        Args:
            encoder_outs: feature dict produced by forward_condition_model.
            t: diffusion timestep.
            event_diffused: ground-truth stage-label embeddings with noise.
        """
        # Ablation hook: the released models condition on the full set of
        # encoder features.
        cond_type = random.choice(self.cond_types)
        if cond_type != "full":
            raise NotImplementedError(f"cond_type '{cond_type}' is not supported.")

        return self.decoder(encoder_outs, t, event_diffused.float())


    def get_training_loss(self, batch,
          sem_encoder_ce_criterion, sem_encoder_mse_criterion, bound_encoder_ce_criterion,
          decoder_ce_criterion, decoder_mse_criterion, decoder_bound_criterion, config):
        
        # video_feats B F C T 
        # event_gt: B C T 

        video_feats = batch["video_feats"].to(self.device) # B F T C 
        boundary_label = batch["boundary_label"].to(self.device) # B T 
        event_gt = batch["f_stage_label"].to(self.device)  # B T
        event_gt_embed = self.f_stage_embed(event_gt.long())
        event_gt_oh = F.one_hot(event_gt.long(), num_classes=self.num_classes) # B T num_classes

        outs = self.forward_condition_model(video_feats, get_features=True)

        sem_encoder_out = outs["sem_encoder_out"]
        sem_encoder_ce_loss = sem_encoder_ce_criterion(
            sem_encoder_out.view(-1, self.num_classes),
            event_gt.view(-1).long()
        )

        # semantic encoder mse loss
        if config["train"]["loss_weights"]["sem_encoder_mse_loss"] > 0.0:
        # # temporal smoothing loss: MSE(P(1: T), P(0: T - 1))
            sem_encoder_mse_loss = torch.clamp(sem_encoder_mse_criterion(
                F.log_softmax(sem_encoder_out[:, 1:, :], dim=2), 
                F.log_softmax(sem_encoder_out.detach()[:, :-1, :], dim=2)), 
                min=0, max=16)
            sem_encoder_mse_loss = sem_encoder_mse_loss.mean()
        else:
            sem_encoder_mse_loss = torch.tensor(0).to(self.device) # No boundary loss for encoder

        # boundary encoder ce loss
        bound_encoder_out = outs["bound_encoder_out"]
        bound_encoder_ce_loss = bound_encoder_ce_criterion(
            bound_encoder_out.view(-1, 2),
            boundary_label.view(-1).long()
        ) 

        # useful for ablation.
        if config["train"]["only_encoder"]:
            loss_dict = {
                'sem_encoder_ce_loss': sem_encoder_ce_loss,
                'sem_encoder_mse_loss': torch.tensor(0).to(self.device),
                'bound_encoder_ce_loss': torch.tensor(0).to(self.device),

                'decoder_ce_loss': torch.tensor(0).to(self.device),
                'decoder_mse_loss': torch.tensor(0).to(self.device),
                'decoder_bound_loss': torch.tensor(0).to(self.device),
            }

            return loss_dict


        ###### Diffusion part ######
        event_diffused, noise, t = self.prepare_targets(event_gt_embed)  # B T embed_dim
        event_out = self.forward(outs, t, event_diffused)

        # Boundary prediction loss for the decoder. It is switched off in every
        # released config; the released recipe only uses the losses below.
        if config["train"]["loss_weights"]["decoder_bound_loss"]:
            raise NotImplementedError(
                "decoder_bound_loss is not part of the released recipe, "
                "please set its weight to 0.0 in the config."
            )
        decoder_bound_loss = torch.tensor(0).to(self.device)

        # compute the frame-wise classification loss for the decoder
        decoder_ce_loss = decoder_ce_criterion(
            event_out.view(-1, self.num_classes),
            event_gt.view(-1).long()
        )

        # compute temporal smooth mse loss for the decoder
        if config["train"]["loss_weights"]["decoder_mse_loss"]:
            decoder_mse_loss = torch.clamp(decoder_mse_criterion(
                F.log_softmax(event_out[:, 1:, :], dim=2), 
                F.log_softmax(event_out.detach()[:, :-1, :], dim=2)), 
                min=0, max=16)
            decoder_mse_loss = decoder_mse_loss.mean()
        else:
            decoder_mse_loss = torch.tensor(0).to(self.device) # No boundary loss for encoder            

        loss_dict = {
            'sem_encoder_ce_loss': sem_encoder_ce_loss,
            'sem_encoder_mse_loss': sem_encoder_mse_loss,
            'bound_encoder_ce_loss': bound_encoder_ce_loss,

            'decoder_ce_loss': decoder_ce_loss,
            'decoder_mse_loss': decoder_mse_loss,
            'decoder_bound_loss': decoder_bound_loss,
        }

        return loss_dict

    @torch.no_grad()
    def ddim_sample(self, video_feats, seed=None, reset_rng_after=False):
        """DDIM sampling of a full stage-label sequence.

        Args:
            video_feats: (B, T, C) or (B, F, T, C) backbone features.
            seed: if given, the RNG is fixed so that sampling is reproducible.
            reset_rng_after: re-seed the global RNG with a time-based seed once
                sampling is finished. Training sets this to True so that the
                training loop does not keep consuming the fixed sampling stream;
                inference keeps the incoming RNG state.
        """
        # video_feats: b f t c
        encoder_outs = self.forward_condition_model(video_feats, get_features=True)

        # NOTE: fix the RNG for each video so that testing is reproducible
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        t_length = video_feats.shape[2] if len(video_feats.shape) == 4 else video_feats.shape[1]
        shape = (video_feats.shape[0], t_length, self.decoder_params["num_f_maps"])
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        # tensor([ -1., 249., 499., 749., 999.])
        times = list(reversed(times.int().tolist()))
        # [999, 749, 499, 249, -1]
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        # [(999, 749), (749, 499), (499, 249), (249, -1)]

        x_time = torch.randn(shape, device=self.device)
        x_start = None
        for time, time_next in time_pairs:

            time_cond = torch.full((1,), time, device=self.device, dtype=torch.long)
            pred_noise, x_start, x_pred = self.model_predictions(encoder_outs, x_time, time_cond)
            x_return = torch.clone(x_pred)
            
            if time_next < 0:
                x_time = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_time)

            x_time = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        # NOTE: re-seed the global RNG after sampling during training
        if seed is not None and reset_rng_after:
            t = 1000 * Time.time() # current time in milliseconds
            t = int(t) % 2**16
            random.seed(t)
            torch.manual_seed(t)
            torch.cuda.manual_seed_all(t)

        return x_return


