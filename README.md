# Testing Attention Refocusing for my master thesis

<div align="center">
<h1>Attention Refocusing</h1>

[Quynh Phung](https://qqphung.github.io/)&nbsp; [Songwei Ge](https://songweige.github.io/)&nbsp; [Jia-Bin Huang](https://jbhuang0604.github.io/)

[[Website](https://attention-refocusing.github.io)][[Demo](https://huggingface.co/spaces/attention-refocusing/Attention-refocusing)]

<h3>Abstract</h3>
Controllable text-to-image synthesis with attention refocusing. We introduce a new framework to improve the controllability of text-to-image synthesis given the text prompts. We first leverage GPT-4 to generate layouts from the text prompts and then use grounded text-to-image methods to generate the images given the layouts and prompts. However, the detailed information, like the quantity, identity, and attributes, is often still incorrect or mixed in the existing models. We propose a training-free method, attention-refocusing, to improve on these aspects substantially. Our method is model-agnostic and can be applied to enhance the control capacity of methods like GLIGEN (top row) and ControlNet (bottom rows)
</div>

## Quick start

```bash
conda create --name attention-refocusing python==3.9
conda activate attention-refocusing
pip install -r requirements.txt
```

Then download the model [GLIGEN](https://huggingface.co/gligen/gligen-generation-text-box/blob/main/diffusion_pytorch_model.bin) and put it in `gligen_checkpoints`

## Image generation

The .csv file containing the prompts should be inside a folder named `prompts` that is posiotioned in the root of the project.

Run the script `inference.py` with the following parameters:
`--folder`: root folder for output (default="results")
`--ckpt`: path to the checkpoint (type=str, default='gligen_checkpoints/diffusion_pytorch_model.bin')
`--batch_size`: (type=int, default=1)
`--guidance_scale`: (type=float, default=7.5)
`--negative_prompt`: (type=str, default='low quality, low res, distortion, watermark, monochrome, cropped, mutation, bad anatomy, collage, border, tiled')
`--file_save`: (default='results', type=str)
`--layout`: (default='layout', type=str)
`--loss_type`: choose one option among the four options for what types of losses (choices=['standard','SAR','CAR','SAR_CAR'],default='SAR_CAR')

NOTE: all the parameters have default values and thus can be omitted.
