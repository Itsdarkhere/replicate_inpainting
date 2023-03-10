import os
from typing import List

import torch
from pytorch_lightning import seed_everything
from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPTextModel, CLIPTokenizer

from PIL import Image
from huggingface_hub import hf_hub_download
from cog import BasePredictor, Input, Path

MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"
MODEL_CACHE = "diffusers-cache"

with open("concepts.txt") as infile:
    CONCEPTS = [line.rstrip() for line in  infile]

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        self.tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_ID,
            subfolder="tokenizer",
            cache_dir="pretrain/tokenizer",
            local_file_only=True,
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            MODEL_ID,
            subfolder="text_encoder",
            cache_dir="pretrain/text_encoder",
            local_files_only=True,
        )

        # self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     MODEL_ID,
        #     cache_dir=MODEL_CACHE,
        #     local_files_only=True,
        # ).to("cuda")

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        # ADDON
        concept: str = Input(
            choices=CONCEPTS,
            default="sd-concepts-library/poolrooms: <poolrooms>",
            description="Choose a pretrained concept. The Placeholder is shown in <your-chosen-concept>."
        ),
        prompt: str = Input(
            description="Input prompt",
            default="a photo of an astronaut riding a horse on mars",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None
        ),
        image: Path = Input(
            description="Inital image to generate variations of. Supproting images size with 512x512",
        ),
        mask: Path = Input(
            description="Black and white image to use as mask for inpainting over the image provided. White pixels are inpainted and black pixels are preserved",
        ),
        num_outputs: int = Input(
            description="Number of images to output. Higher number of outputs may OOM.",
            ge=1,
            le=8,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=25
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # ADDON
        seed_everything(seed)

        # ADDON
        repo_id_embeds = concept.split(":")[0]
        print(f"Using concept: {repo_id_embeds}")
        embeds_path = hf_hub_download(
            repo_id=repo_id_embeds,
            filename="learned_embeds.bin",
            cache_dir=repo_id_embeds,
            local_file_only=True,
        )
        print(f"Embeds path: {embeds_path}")
        token_path = hf_hub_download(
            repo_id=repo_id_embeds,
            filename="token_identifier.bin",
            cache_dir=repo_id_embeds,
            local_file_only=True,
        )

        # ADDON
        with open(token_path, "r") as file:
            placeholder = file.read()

        print(f"Placeholder: {placeholder}")

        # ADDON
        loaded_learned_embeds = torch.load(embeds_path, map_location="cpu")

        # ADDON, separate token and embeds
        trained_token = list(loaded_learned_embeds.keys())[0]
        embeds = loaded_learned_embeds[trained_token]

        # ADDON, cast to dtype of text_encoder
        dtype = self.text_encoder.get_input_embeddings().weight.dtype
        embeds.to(dtype)

        # ADDON, add the token in tokenizer
        num_added_tokens = self.tokenizer.add_tokens(trained_token)
        print(f"New tokens: {num_added_tokens}")

        # ADDON, resize the token embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # ADDON, get the id for the token and assign the embeds
        token_id = self.tokenizer.convert_tokens_to_ids(trained_token)
        self.text_encoder.get_input_embeddings().weight.data[token_id] = embeds

        # ADDON
        print("Loading pipeline with added tokenizer and text_encoder...")
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            torch_dtype=torch.float16,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
        ).to("cuda")

        image = Image.open(image).convert("RGB").resize((512, 512))
        extra_kwargs = {
            "mask_image": Image.open(mask).convert("RGB").resize(image.size),
            "image": image
        }

        # REMOVAL
        # generator = torch.Generator("cuda").manual_seed(seed)

        print("Generating images with the learned concept") 
        with torch.autocast("cuda"):
            images = pipeline(
                prompt=[prompt] * num_outputs if prompt is not None else None,
                negative_prompt=[negative_prompt] * num_outputs
                if negative_prompt is not None
                else None,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                **extra_kwargs,
            )["sample"]

        # REMOVAL
        # output = self.pipe(
        #     prompt=[prompt] * num_outputs if prompt is not None else None,
        #     negative_prompt=[negative_prompt] * num_outputs
        #     if negative_prompt is not None
        #     else None,
        #     guidance_scale=guidance_scale,
        #     generator=generator,
        #     num_inference_steps=num_inference_steps,
        #     **extra_kwargs,
        # )

        output_paths = []
        for i, image in enumerate(images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
