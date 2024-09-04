

from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from util.attn import AttnCLusterProcessor

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from copy import deepcopy


class StableDiffusion(nn.Module):
    def __init__(
            self,
            sd_version="2.0",
            half=True,
            step_guidance=None,
            attention_layers_to_use=[]

    ):
        super().__init__()

        self.sd_version = sd_version
        self.use_half = half
        print(f"[INFO] loading stable diffusion...")

        model_key = "stabilityai/stable-diffusion-2-1-base"

        # Create model

        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae")
        self.vae = self.vae.half() if self.use_half else self.vae
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder")
        self.text_encoder = self.text_encoder.half() if self.use_half else self.text_encoder
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet")
        self.unet = self.unet.half() if self.use_half else self.unet

        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False

        self.vae.eval()
        self.text_encoder.eval()
        self.unet.eval()
        # del self.vae.decoder
        self.scheduler = DDIMScheduler.from_config(model_key, subfolder="scheduler")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.020)
        self.max_step = int(self.num_train_timesteps * 0.980)
        if step_guidance is not None:
            self.min_step, self.max_step = step_guidance

        self.alphas = self.scheduler.alphas_cumprod  # for convenience
        self.device = None
        self.device1 = None
        print(f"[INFO] loaded stable diffusion!")

        self.noise = None

        # 获取部分中间结果
        self.attention_maps = {}

        def create_nested_hook_for_attention_modules(n):
            def hook(module, input, output):
                bs_head, h, w = output[1].shape
                self.attention_maps[n] = output[1].reshape(bs_head // module.heads, module.heads, h, w)

            return hook

        self.handles = []
        self.processor = AttnCLusterProcessor()
        for module in attention_layers_to_use:
            self.handles.append(
                eval("self.unet." + module).register_forward_hook(
                    create_nested_hook_for_attention_modules(module)
                )
            )
            eval("self.unet." + module).processor = self.processor

    def setup(self, device, device1=None):
        self.device1 = device if device1 is None else device1
        self.vae = self.vae.to(device)
        self.unet = self.unet.to(self.device1)
        self.text_encoder = self.text_encoder.to(device)
        self.alphas = self.alphas.to(device)
        self.device = device

    def get_text_embeds(self, prompt, negative_prompt, **kwargs):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.set_grad_enabled(False):
            text_embeddings = self.text_encoder(text_input.input_ids)[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        with torch.set_grad_enabled(False):
            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]

        return uncond_embeddings, text_embeddings

    def get_attention_map(self):
        raw_attention_maps = self.attention_maps
        cross_attention_maps = {8: [], 16: [], 32: [], 64: []}
        self_attention_maps = {8: [], 16: [], 32: [], 64: []}

        # 同尺寸层次归类
        for layer in raw_attention_maps:
            bs, head, img_embed_len, text_embed_len = raw_attention_maps[layer].shape
            hw = int(math.sqrt(img_embed_len))
            reshaped_attn = raw_attention_maps[layer].reshape(bs, head, hw, hw, text_embed_len).softmax(-1)
            if layer.endswith("attn2"):  # cross attentions
                cross_attention_maps[hw].append(reshaped_attn)
            elif layer.endswith("attn1"):  # self attentions
                self_attention_maps[hw].append(reshaped_attn)
        for key, values in cross_attention_maps.items():
            if len(values) == 0: continue
            attn = torch.cat(values, dim=1)
            attn = attn.permute(0, 1, 4, 2, 3)
            cross_attention_maps[key] = attn

        for key, values in self_attention_maps.items():
            if len(values) == 0: continue
            attn = torch.cat(values, dim=1)
            attn = attn.permute(0, 1, 4, 2, 3)
            self_attention_maps[key] = attn
        return cross_attention_maps, self_attention_maps

    def train_step(
            self,
            text_embeddings,
            input_image,
            guidance_scale=100,
            t=None,
            generate_new_noise=True,
            attention_map=None,
            latents=None
    ):
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')
        if not (input_image.shape[-2] == 512 and input_image.shape[-1] == 512):
            input_image = F.interpolate(
                input_image, (512, 512), mode="bilinear", align_corners=False
            )
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if len(t) == 2:
            t = torch.randint(t[0], t[1] + 1, [1], dtype=torch.long)
        t = t.to(self.device)
        # encode image into latents with vae, requires grad!
        # _t = time.time()
        if latents is None:
            latents = self.encode_imgs(input_image)
        if attention_map is not None:
            latents = latents * attention_map.to(self.device)
        # _t = time.time()
        with torch.set_grad_enabled(True):
            # add noise
            if generate_new_noise is not None:
                if hasattr(self, "rand_seed"):
                    torch.cuda.manual_seed(self.rand_seed)
                    torch.manual_seed(self.rand_seed)
                noise = torch.randn_like(latents).to(self.device)
                self.noise = noise
            else:
                noise = self.noise.to(self.device)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latents_noisy = torch.cat([latents_noisy] * len(text_embeddings), dim=0)
            # text_embeddings_neg = text_embeddings.clone()
            # text_embeddings_neg[:, 4, :] = -text_embeddings_neg[:, 4, :]
            # text_embeddings = torch.cat([text_embeddings, self.te], dim=0)
            # pred noise
            noise_pred_ = self.unet(
                latents_noisy.to(self.device1),
                t.to(self.device1),
                encoder_hidden_states=text_embeddings.to(self.device1),
            ).sample.to(self.device)
        self.scheduler.set_timesteps(1000)
        if self.processor.mask is not None:
            latents = self.scheduler.step(noise_pred_.cpu(), 1, latents_noisy.cpu())["pred_original_sample"].cuda()
            img = self.decode_latents(latents)
            import matplotlib.pyplot as plt
            plt.imshow(img[0].permute(1, 2, 0).float().cpu())
            return None
        else:
            cross_attention_maps, self_attention_maps = self.get_attention_map()
            return cross_attention_maps, self_attention_maps

    def produce_latents(
            self,
            text_embeddings,
            height=512,
            width=512,
            num_inference_steps=50,
            guidance_scale=7.5,
            latents=None,
    ):
        if latents is None:
            latents = torch.randn(
                (
                    text_embeddings.shape[0] // 2,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )

        self.scheduler.set_timesteps(num_inference_steps)
        all_attention_maps = []

        with torch.autocast("cuda"):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input.to(self.device1),
                        t.to(self.device1),
                        encoder_hidden_states=text_embeddings.to(self.device1),
                    )["sample"].to(self.device)

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
                if 10 <= i <= 25:
                    all_attention_maps.append(deepcopy(self.attention_maps))

        return latents, all_attention_maps

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        # latents = posterior.sample() * 0.18215
        latents = posterior.mean * 0.18215
        return latents

    def prompt_to_img(
            self,
            prompts,
            negative_prompts="",
            height=512,
            width=512,
            num_inference_steps=50,
            guidance_scale=7.5,
            latents=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]
        text_embeds = torch.cat(text_embeds, dim=0)
        # Text embeds -> img latents
        latents, all_attention_maps = self.produce_latents(
            text_embeds,
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs, all_attention_maps
