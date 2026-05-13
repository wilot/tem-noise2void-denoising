"""swinunet

An implementation of a Swin-UNet.

The SwinUNet is broadly a UNet but using double Swin transformer layers rather than double 2D convolutional layers. The
Swin Transformer layers are a modification on a classical multi-head self-attention layer that might be found in ViT
for example. Rather than computing the self-attention from each patch/token to every other patch, patches are grouped
into 'windows' and attention is only calculated within each window. This converts the attention problem from scaling
quadratically with image size to a problem scaling linearly (for constant window-size). After each layer the window
boundaries are shifted by half the window size (in units of patches) to allow information/attention to cross the
window boundaries.

A series of abbreviations are used in the comments to help describe the changes to dimensionality. H, W are height
and width of the input image, in pixels. H_p, W_p are the number of patches in the vertical and horizontal directions,
within the image, for a total of P patches. H_w, W_w are the number of windows in the image. Therefore, each patch is
H // H_p pixels vertically, each window is H_p // H_w patches vertically and each window is H // H_w pixels vertically.
The input image has C image channels. The embedding dimension of the patches is E.

Both the window size and the batch size need to be divisible by the image size!

Modifications to the paper:
- In each layer, the spatial dimensions are reduced or expanded by a factor of four (matching the patch size). Here, a
  factor of two is used while the patch size remains four.
- Instead of the final patch expanding layer in the decoder, a bilinear interpolation is used along with a 1x1
  convolution, to convert the patches back into pixels.
- The paper was a bit ambiguous as to where the skip-connection is concatenated in the decoder layers. Here the decoder
  layers first upsample the patches from the previous layer, then concatenate (in the channel axis) the skip connection,
  then pass through a FCN to reduce the embedding dimension, then pass through SwinTransformerBlocks.
"""
from typing import Callable, Optional

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Split image into patches and embed them."""

    def __init__(self, img_size=1024, patch_size=4, in_chans=1, embed_dim=16):
        super().__init__()
        assert img_size % patch_size == 0  # Ensure divisible
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = img_size // patch_size
        self.num_patches = self.patches_resolution ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=2 * patch_size, stride=patch_size, padding=2)
        self.norm = nn.LayerNorm(embed_dim)  # TODO: Necessary?

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)  # B, E, H_p, W_p
        x = x.flatten(2).transpose(1, 2)  # B, P, E
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    """Merge 2x2 patches (downsampling) while doubling the embedding dimension."""

    SCALE = 2  # The scale of the downsampling. H_p & W_p are halved, so the number of patches is quartered.

    def __init__(self, embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed_dim = embed_dim
        self.reduction = nn.Linear(4 * embed_dim, 2 * embed_dim, bias=False)
        self.norm = norm_layer(4 * embed_dim)  # TODO: Necessary?

    def forward(self, x: torch.Tensor, H_p: int, W_p: int) -> torch.Tensor:
        B, P, E = x.shape
        assert P == H_p * W_p, f"input feature has wrong size: {P=} {H_p=} {W_p=}"
        x = x.view(B, H_p, W_p, E)

        # Fold the patches so that the four patches contributing to each new patch in the 2x downsampled patches
        # are spatially colocated but stacked in the embed-dimension, then reduce the embedding dimension.

        x0 = x[:, 0::2, 0::2, :]  # even rows & cols patches
        x1 = x[:, 1::2, 0::2, :]  # odd rows, even cols patches
        x2 = x[:, 0::2, 1::2, :]  # even rows, odd cols patches
        x3 = x[:, 1::2, 1::2, :]  # odd rows, odd cols patches
        x = torch.cat([x0, x1, x2, x3], -1)  # batch, H_patches/2, W_patches/2, 4*E
        x = x.view(B, -1, 4 * E)  # B, P // 4, 4*E

        x = self.norm(x)
        x = self.reduction(x)  # B, P // 4, 2 * E

        return x


class PatchExpanding(nn.Module):
    """Expand patches (upsampling) while halving the embedding dimension."""

    def __init__(self, embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed_dim = embed_dim
        self.expand = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.norm = norm_layer(embed_dim // 2)

    def forward(self, x: torch.Tensor, H_p, W_p):
        B, P, E = x.shape
        assert P == H_p * W_p, f"input feature has wrong size: {P=} {H_p=} {W_p=}"
        x = self.expand(x)  # Doubles the channels

        # Pixel shuffle works by dividing the channels by 4 and doubling H_p and W_p by rearranging the axes in memory

        x = x.view(B, H_p, W_p, 2 * E)
        x = torch.nn.functional.pixel_shuffle(x.permute(0, 3, 1, 2), 2)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, E // 2)
        x = self.norm(x)

        return x


def window_partition(x: torch.Tensor, window_size: int):
    """Partition patch-grid into non-overlapping windows. Requires tailing channels. Window-size is side-length and
    H_p, W_p should each be divisible by window_size. Takes a (B, H_p, W_p, E) tensor and returns a
    (B * H_w * W_w, H_p / H_w, W_p / W_w, E) for windows of size (H_p / H_w, W_p / W_w) patches."""
    B, H_p, W_p, E = x.shape
    x = x.view(B, H_p // window_size, window_size, W_p // window_size, window_size, E)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, E)
    return windows


def window_reverse(windows, window_size: int, H_p: int, W_p: int):
    """Reverse window partition. The images are grouped into patches. This patched-image is the grouped into windows,
    within which attention is performed. This function takes a stack of windows, each with P patches. It converts
    that back into the patch image. Input is of shape (nWindows, H_p // H_w, W_p // W_w, E) and returns
    (batch-images, H_p, W_p, E). H_p is the height of the image, in patches. window_size is the size of each window, in
    patches."""

    assert H_p % window_size == 0 and W_p % window_size == 0, "The number of patches is not divisible by window-size!"

    H_w, W_w = H_p // window_size, W_p // window_size
    windows_per_image = H_w * W_w
    B = int(windows.shape[0] / windows_per_image)
    x = windows.view(B, H_w, W_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_p, W_p, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self attention."""

    def __init__(self, embed_dim, window_size, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "The embedded dimensions must be divisible by the number of attention heads"
        self.embed_dim = embed_dim
        self.window_size = window_size  # Window side-length in units of patches
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.inv_root_d = head_dim ** -0.5  # The scale factor from the paper

        # Learnable relative position bias
        # We look up the relative offset between two patch indices in a window. This relative offset is the key for
        # a learned bias term.
        # A constant lookup table for relative offset (Manhattan) from a pair of patch indices is pre-calculated. This
        # is shaped (M^2, M^2, 2) with the last index for the height/width axis. Its values are initially ranged
        # (-M+1, M-1). From this M-1 is added to make non-negative (for indexing) making it valued (0, 2M-2). This
        # offset value is then converted to a linear (C array) index for a (2M-1)^2 flat array by multiplying the Y
        # offset values by 2M-1. Then the Y & X offsets are added, making `relative_position_index` a (M^2, M^2) array
        # for indexing a ((2M-1)^2,) array. At runtime, the learned ((2M-1)^2, num_heads) bias array, which contains
        # a bias value for every *UNIQUE* relative offset, is indexed with every offset from the (non-unique) (M^2, M^2)
        # array. This way, a bias parameter is learned only for every unique offset, despite the relative offset between
        # any two patch parings being non-unique.

        self.relative_position_bias_table = nn.Parameter(  # The (2M-1 * 2M-1, num_heads) learned bias table, one for each MSA head
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords_h = torch.arange(self.window_size)  # Window size is side-length of window in number of patches
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # The (2, M^2, M^2) full YX offset grid
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (M^2, M^2, 2) lookup table
        relative_coords[:, :, 0] += self.window_size - 1  # Add M-1 to make non-negative
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1  # Since each step in height is skipping a whole row
        relative_position_index = relative_coords.sum(-1)  # Sum the vertical & horizontal offsets to get linear (in memory) distance
        self.register_buffer("relative_position_index", relative_position_index)  # The index-pair -> relative offset lookup table

        # Project into query, key, value. The embedded dims is be equally split across the attention heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """Shifted window multi-head self attention. The embedding dimension is shared equally across the attention
        heads and therefore should be divisible by the number of attention heads. The window-shifting should already
        have been performed if necessary. If it has, the shifted-window mask must be supplied.

        This takes a stack of windows x (shape windows-per-image * images-per-batch, patches-per-window, embedding dim)
        and processes it. First it is passed through a FCN which triples the embedding dimension, which is split into
        the query, key and mask matrices. These are split (in the embedding dimension) equally across the attention
        heads. Then softmax(cyclic-swin-mask(q * k.T * 1/sqrt(d) + b)) is computed as per the SwinTransformer paper,
        where the bias is the learned bias for every relative offset between patches.

        Then the shifting windows are applied. The shift is performed cyclically. To ensure that attention cannot also
        happen cyclically, windows which contain wrapped-around patches are masked. Attention values from a patch to
        another patch that would cross the wrap-around boundary are given a -100. bias using the mask. The mask is in
        shape (windows-per-image, patches-per-window, patches-per-window).

        Returns a tensor of the same shape as x, should be number of windows (across the whole batch), number of patches
        in the window, embedding dimension.
        """

        B, P, E = x.shape  # Where here B is the total number of windows in the whole batch (many windows per image)
        qkv = self.qkv(x).reshape(B, P, 3, self.num_heads, E // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each of shape (B, heads, patches, embed-dim)
        print(f"\t\tWindow Attention: {q.shape=} {k.shape=} {v.shape=}")

        q = q * self.inv_root_d
        attn = (q @ k.transpose(-2, -1))  # Shape (B, heads, patches, patches) is patch-to-patch attention within each window

        # Look up relative offset, then use that to look up learned bias
        # relative_position_index.view(-1) is (patches * patches,) table of the relative offset value for every combination of patches
        # relative_position_bias_table[the above] is the (patches * patches, num_heads) table of learned bias values
        # the above.view(self.window_size * ... -1) is the learned bias for every relative index pair, per head.
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)] \
            .view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)  # B, heads, query_patch_index, key_patch_index
        print(f"\t\tWindowAttention: {attn.shape=}")

        if mask is not None:
            nW = mask.shape[0]  # Total number of windows (includes the whole batch)
            attn = attn.view(B // nW, nW, self.num_heads, P, P) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, P, P)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v)  # .transpose(1, 2)
        print(f"\t\tWindowAttention: attention shape {x.shape=}")
        x = x.transpose(1, 2).reshape(B, P, E)
        print(f"\t\tWindowAttention: attention reshaped to {x.shape=}")
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""

    def __init__(self, embed_dim, num_heads, window_size=32, shift_size=0, mlp_ratio=4.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size  # In units of patches
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = WindowAttention(embed_dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )

    def forward(self, x, H_p, W_p):
        B, P, E = x.shape
        assert P == H_p * W_p, "input feature has wrong size"
        assert H_p % self.window_size == 0 and W_p % self.window_size == 0, "non-integer number of windows"

        residual = x  # For residual skip connection *within* the Swin block
        x = self.norm1(x)
        x = x.view(B, H_p, W_p, E)

        print(f"\tSwinTransformerBlock: {B=} {H_p=} {W_p=} {E=}")
        print(f"\tSwinTransformerBlock: {self.shift_size=}")

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self.create_mask(H_p, W_p).to(x.device)  # Shape windows_per_image, patches_per_window, patches_per_window
            print(f"\tSwinTransformerBlock: {attn_mask.shape=}")
        else:
            shifted_x = x
            attn_mask = None

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)  # Now shape nWindows * B, H_p // H_w, W_p // W_w, E
        print(f"\tSwinTransformerBlock: Windowed input {x_windows.shape=}")
        x_windows = x_windows.view(-1, self.window_size * self.window_size, E)  # Shape total_windows, patches_per_window, E

        # Attention
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Departition windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, E)
        shifted_x = window_reverse(attn_windows, self.window_size, H_p, W_p)  # Shape B, H_p, W_p, E
        print(f"\tSwinTransformerBlock: De-windowed output shape {shifted_x.shape=}")

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Residual and fully-connected section
        x = residual + x.reshape(B, P, E)  # Residual across the swin-msa
        x = x + self.mlp(self.norm2(x))  # Residual across the FCN/MLP

        return x

    def create_mask(self, H_p, W_p):
        """Creates the shifted window attention mask to ensure that, when shifting the windows cyclically, attention
        is not allowed to wrap around to the other side of the image unintentionally.

        On every other SwinTransformer layer, the windows are shifted by half a window-width to allow attention to
        spread across the image. At the borders this is wrapped around. However, care must be taken to ensure that
        attention cannot unduly wrap around as well.

        A mask/grid of all the patches in the image (one gridpoint per patch) is created. The upper left corner is treated
        normally but the lower right sides are not. Patches that are within two half-window lengths of these sides are
        marked with a unique index (see `count`). The mask is then chunked into a stack of windows. This stack of
        windows (shape nWindows, patches in the window in Y, X) is flattened so that, for each window, a grid point is
        made for every pairing of two patches within that window. At each pairing in this attn_mask, the `count` index
        is subtracted. Regions that should not be able to attent to one another (and thus have non-zero difference) are
        given a value of -100.0

        The final shape of the attention mask is (num_windows per image, patches_per_window, patches_per_window).
        """

        # First create a mask of size H_p, W_p (the number of patches in the image in Y&X)
        img_mask = torch.zeros((1, H_p, W_p, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = count
                count += 1

        mask_windows = window_partition(img_mask, self.window_size)  # To shape nWindows, H_p // H_w, W_p // W_w
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # Flatten to a stack of patches per window
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # Difference for every combination of patches
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


class EncoderLayer(nn.Module):
    """A basic Swin Transformer layer."""

    def __init__(self, embed_dim, depth, num_heads, window_size, shift_windows=True, downsample=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth  # The number of Swin Transformer blocks in the layer
        self.window_size = window_size

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) or not shift_windows else window_size // 2  # Shift window every other block
            )
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample: PatchMerging = downsample(embed_dim=embed_dim)
        else:
            self.downsample = None

    def forward(self, x, H_p, W_p):
        for blk in self.blocks:
            x = blk(x, H_p, W_p)

        if self.downsample is not None:
            x = self.downsample(x, H_p, W_p)
            H_p_output, W_p_output = (H_p + 1) // 2, (W_p + 1) // 2
        else:
            H_p_output, W_p_output = H_p, W_p

        return x, H_p_output, W_p_output


class DecoderLayer(nn.Module):
    """A basic Swin Transformer layer for upsampling. Contains the skip layer FCN.

    The decoder consists of a patch-expanding block, which doubles the patch-resolution (quadruples the number of
    patches) while halving the embedding dimension. The input is then concatenated, in the embedding dimension, with the
    skip connection and the result is passed through a FCN to half the embedding dimension. The resultant embedding
    dimension of the patches is half the input. This result is then passed through a pair of `SwinTransformerBlock`s.
    """

    def __init__(
        self, embed_dim, depth, num_heads, window_size, shift_windows=True,
        upsample=Callable[[torch.Tensor, int, int, int], torch.Tensor] | None
    ):
        """
        Parameters:
        :param embed_dim: The embedding dimension at the input (before patch-expanding)
        :param depth: The number of SwinTransformerBlock layers in the decoder
        :param num_heads: The number of attention heads in the MSA blocks
        :param window_size: The size of the Swin windows, in patches, side-length
        :param upsample: A function to upsample the patch-image
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth

        # Quadruples the number of patches while halving their embedding dimension
        if upsample is not None:
            self.upsample: PatchExpanding = upsample(embed_dim=embed_dim)
        else:
            self.upsample = None

        # After concatenating with the skip dimension (in the embedding axis) the patch embedding dimension equals
        # `embedding_dim`. This is reduced back to `embedding_dim // 2` using a FCN.
        self.concat_linear = nn.Linear(embed_dim, embed_dim // 2)

        # After patch expanding, concatenation & FCN, the embedding dimension is half the input
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                embed_dim=embed_dim // 2,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) or not shift_windows else window_size // 2
            )
            for i in range(depth)
        ])

    def forward(self, x, H_p, W_p, skip: torch.Tensor):

        if self.upsample is not None:
            x = self.upsample(x, H_p, W_p)
            H_p_output, W_p_output = H_p * 2, W_p * 2
        else:
            H_p_output, W_p_output = H_p, W_p

        x = torch.cat([x, skip], -1)  # Incorporate the skip connection (doubles the channels)
        x = self.concat_linear(x)  # Half the channels again

        for blk in self.blocks:
            x = blk(x, H_p_output, W_p_output)

        return x, H_p_output, W_p_output


class SwinTransformer(nn.Module):
    """
    Swin-Transformer: A transformer architecture for image problems using shifted window attention.


    """

    def __init__(
        self, img_size=512, patch_size=4, in_chans=1, out_chans=1, num_layers=4, embed_dim=32, window_size=32,
        num_heads=[4, 8, 8, 8], mlp_ratio=4.0
    ):
        super().__init__()

        assert img_size % patch_size == 0, "The patch size is not divisible by the image size."
        assert img_size % (window_size * patch_size) == 0, "The window size is not divisible by the image size."
        assert len(num_heads) == num_layers, "The number of layers does not match the length of the list of attention heads."

        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.layers = nn.ModuleList([
            SwinTransformerBlock(
                embed_dim, num_heads[layer_index], window_size,
                shift_size=0 if (layer_index % 2 == 0) else window_size // 2,  # Shift window every other block
                mlp_ratio=mlp_ratio
            ) for layer_index in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.up = nn.PixelShuffle(patch_size)
        self.output = nn.Conv2d(embed_dim // (patch_size ** 2), out_chans, kernel_size=1)

        self.apply(self._init_weights)

    def forward(self, x):

        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)
        H_p, W_p = H // self.patch_size, W // self.patch_size

        for layer in self.layers:
            x = layer(x, H_p, W_p)

        x = self.norm(x)
        x = x.view(B, H_p, W_p, -1).permute(0, 3, 1, 2)

        # Final upsampling to original resolution
        x = self.up(x)
        x = self.output(x)

        return x

    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class SwinUNet(nn.Module):
    """
    Swin-UNet: Swin Transformer based U-Net for image segmentation/regression.

    This implementation only works for square images of a power of two in size. Due to patch merging in the network,
    the size of the image (in patches) could become less than the window size. If this happens, the window size is
    reduced so that the whole image fits within a single window.

    Args:
        img_size: Input image size
        patch_size: Patch size for patch embedding
        in_chans: Number of input channels
        out_chans: Number of output channels
        embed_dim: Base embedding dimension
        depths: Depth of each Swin Transformer layer
        num_heads: Number of attention heads in each layer
        window_size: Window size for window attention
    """

    def __init__(self, img_size=512, patch_size=4, in_chans=1, out_chans=1,
                 embed_dim=16, depths=[2, 2, 2, 2], num_heads=[4, 8, 8, 8],
                 window_size=32):
        super().__init__()

        assert img_size % patch_size == 0, "The patch size is not divisible by the image size."
        assert img_size % (window_size * patch_size) == 0, "The window size is not divisible by the image size."
        assert (img_size // patch_size // 2 ** (len(depths) - 1)) >= window_size, "Window size is bigger than entire image after patch merging"

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim
        )

        img_size_p = img_size // patch_size  # The initial size of the image, in patches

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = EncoderLayer(
                embed_dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=min(window_size, img_size_p // 2 ** i_layer),  # Window no bigger than the max num of patches
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
            )
            self.encoder_layers.append(layer)

        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            layer_up = DecoderLayer(
                embed_dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                depth=depths[self.num_layers - 1 - i_layer],
                num_heads=num_heads[self.num_layers - 1 - i_layer],
                window_size=min(window_size, img_size_p // 2 ** (self.num_layers - i_layer - 2)),
                upsample=PatchExpanding
            )
            self.decoder_layers.append(layer_up)

        # self.norm = nn.LayerNorm(embed_dim)

        # Convert patches back into pixels
        # self.up = nn.ConvTranspose2d(embed_dim, embed_dim // patch_size, patch_size, patch_size)
        # self.output = nn.Conv2d(embed_dim // patch_size, out_chans, 1)
        # self.up = nn.Upsample(scale_factor=patch_size, mode="bilinear")
        # self.output = nn.Conv2d(embed_dim, out_chans, kernel_size=1)
        self.up = nn.PixelShuffle(patch_size)
        self.output = nn.Conv2d(embed_dim // (patch_size ** 2), out_chans, kernel_size=1)

        self.apply(self._init_weights)

    def forward(self, x):

        print("SwinUNet forward pass")
        B, C, H, W = x.shape
        print(f"x shape: {B=} {C=} {H=} {W=}")

        # Patch embedding
        x = self.patch_embed(x)
        print(f"embedded x shape: {x.shape=}")
        H_p, W_p = H // self.patch_size, W // self.patch_size
        print(f"After patch embedding {H_p=} {W_p=}\n")

        # Encoder
        x_downsample = []
        for layer in self.encoder_layers:
            print("\nEncoder layer")
            print(f"Encoder window size: {layer.window_size=}")
            x_downsample.append(x)
            x, H_p, W_p = layer(x, H_p, W_p)
            print(f"Encoder layer output {H_p=} {W_p=} {x.shape=}")

        # Decoder
        for i, decoder_layer in enumerate(self.decoder_layers):
            print("\nDecoder layer")
            x, H_p, W_p = decoder_layer(x, H_p, W_p, x_downsample[self.num_layers - 2 - i])
            print(f"Decoded output shape {x.shape=}")
            print(f"Decoder output {H_p=} {W_p=}")

        # x = self.norm(x)
        x = x.view(B, H_p, W_p, -1).permute(0, 3, 1, 2)

        # Final upsampling to original resolution
        x = self.up(x)
        x = self.output(x)

        return x

    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


if __name__ == "__main__":

    # For an image size of 1024x1024 dual-channel, patch size of 4, there are 256x256 patches in the image
    # These are embedded from inherent 32 dimension down to 16
    # Window size of 64x64 patches gives 4x4 windows per image, each window equivalent to 256x256px image region at input
    model = SwinUNet(
        img_size=512, patch_size=4, in_chans=1, out_chans=1, embed_dim=32, depths=[2, 2, 2, 2], num_heads=[4, 8, 8, 8],
        window_size=16
    )
    model = model.to(device=torch.device("cuda:0"))
    image = torch.ones(8, 1, 512, 512, device=torch.device("cuda:0"))
    output = model(image)
    print(f"{image.shape=}")
    print(f"{output.shape=}")