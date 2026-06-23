# Akatz Seamless Tiling

ComfyUI custom nodes for seamless tiling workflows and small Akatz workflow
utilities.

## Nodes

- `AK_SeamlessTile`
- `AK_PeriodicLatentConstraint`
- `AK_CircularVAEDecode`
- `AK_MakeCircularVAE`
- `AK_OffsetImage`
- `AK_SeamRepairPrepare`
- `AK_SeamRepairFinish`
- `AK_MaskedLatentNoise`
- `AK_FileNamePrefixDateDirFirst`
- `AK_MakeDepthmapSeamless`
- `AK_MakeDepthmapSeamlessAdvanced`
- `AK_TileStripJoin`
- `AK_TileStripWindow`
- `AK_AnimatedDilationMaskLinear`
- `AnySwitch | Akatz`
- `AkatzSetNode` / `AkatzGetNode` frontend virtual nodes

The seamless model node uses ComfyUI's model patcher clone API instead of
Python `deepcopy`, which avoids failures with current dynamic model patchers.
`AK_PeriodicLatentConstraint` is an experimental sampler patch for transformer
models such as Z-Image where convolution padding patches do not affect the main
denoising path.
`AK_SeamRepairPrepare` and `AK_SeamRepairFinish` support offset-and-inpaint
seam repair: move wrap seams into the image interior, repair the masked seam,
then composite and restore the tile back to its original alignment.
`AK_MaskedLatentNoise` can inject deterministic fresh noise into a masked latent
band so an img2img repair sampler has enough signal to repaint the seam.

Some workflows that use this pack may still require separate node packs such as
ComfyUI Depthflow Nodes, ComfyUI Basic Math, and ComfyUI DepthAnythingV2.

## License

GPL-3.0. See `LICENSE`.

This pack consolidates and adapts behavior from:

- `ComfyUI-seamless-tiling` by spinagon, GPL-3.0.
- `deforum-comfy-nodes` by deforum-art, MIT, for the Any Switch and frontend
  virtual Set/Get node patterns.
- `ComfyUI-AKatz-Nodes` by akatz, MIT, for the Akatz utility nodes.
