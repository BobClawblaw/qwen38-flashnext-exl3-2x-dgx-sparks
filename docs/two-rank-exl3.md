# EXL3 on two ranks: how the pack is split

TensorFold's two-rank Flash Next engine was written for the MLX 4-bit checkpoint, where a rank's share of a matrix is a slice of rows or of 32-wide input groups. An EXL3 matrix is trellis-coded in 16x16 tiles in a rotated domain: a 128-wide Hadamard transform is applied to every block of 128 inputs and of 128 outputs before quantization, and the kernel undoes it on the activations. A share must therefore keep whole tiles and whole 128-blocks. This note records how the recipe's patch (part 6) does that and why the experts are shared differently.

## Dense projections

With `H` the 128-block Hadamard scaled by `1/sqrt(128)`:

```
y = ((x * suh) @ H_K @ W_q) @ H_N * svh + bias
```

- **By outputs** (q with its gates, k, v, the DeltaNet q/k/v and z projections, the head): take the columns of tiles `[n0/16, n1/16)` and `svh[n0:n1]`, all of `suh`. `H_N` is block-diagonal, so a 128-aligned slice of outputs depends on nothing outside it.
- **By inputs** (o_proj, out_proj): take the rows of tiles `[k0/16, k1/16)` and `suh[k0:k1]`, all of `svh`; each rank's output is a partial sum and the ranks add them (fp32, gathered in rank order, as the MLX path does). The bias is added on one rank.

Flash Next's shapes all divide: 24 heads x 256 (q, with gates: 12,288 rows), 2 KV heads (512), 16 key and 48 value DeltaNet heads x 128, and a 248,320-row head (970 blocks a rank). `exl3_mm.x3_part` performs the cut on the CPU copy of the tensors before the strips layout is built, so a rank never holds the other rank's half.

The one wrinkle was in the engine, not the format: `_out_proj` returned a non-MLX projection's output as a finished branch even on two ranks, so each rank kept only its own heads. It now gathers fp32 partials for any weight type when a communicator is present (the EXL3 linear writes fp32).

## Experts

A routed expert is 640 wide: five Hadamard blocks, which two ranks cannot halve, and the grouped EXL3 kernel takes one width a layer. Three options were weighed: pad both halves to 384 (a fifth of the expert bandwidth wasted), split 384/256 and alternate the big half expert by expert (needs a width per expert in the kernel), or hold whole experts. The recipe holds whole experts:

- rank 0 holds experts 0 to 255, rank 1 holds 256 to 511, and the shared expert alternates ranks layer by layer, so both ranks carry the same expert bytes over the model;
- both ranks run the router and pick the same ten experts; `experts.share` sends a window's picks through the layer's `remap`, which maps an expert held here to its local row and any other to the local count, which every grouped kernel skips, and zeroes that slot's weight;
- `routed(..., wts, out=partial)` then produces this rank's weighted sum over the experts it holds, and the two partials are gathered and added exactly as the MLX path adds its half-width partials.

The weight zeroing matters: the fused down-and-combine kernel takes a skipped slot's output from `y` as the caller left it (GLM's path fills such slots itself), and the engine keeps one scratch for every layer, so without it a skipped slot would add the previous layer's output. The unit test runs the shares twice through one scratch for that reason.

Per token a rank runs the experts it holds: five of ten on average, up to ten. For one stream the slower rank sets the round, about a quarter over a perfect split in the worst case; at four rows a round or more it averages out, and at sixteen users it is noise. That, and the EXL3 expert kernel's cost at many rows (it decodes a tile again for every 16 rows), is why this recipe is slower than the MLX pair at one user and at eight, and ahead of it on structured replies at sixteen.

## Everything else

Hyper-connections, the n-gram branch, the embedding and the mixer are replicated (small, fp16 or bf16). The MTP head's attention and experts split like the main layers'; its two fc projections are replicated. Each rank maps the pack's n-gram table from its own copy of the file. The draft head's rows (the MTP drafts' vocabulary subset, split between the ranks as the MLX path splits it) are decoded from the whole head once at load, then the whole head is dropped. The two-rank vision exchange, structured-output refusal and candidate gathering are format-agnostic and unchanged.

## Measured

Two ranks against one GPU, the same pack, greedy, six prompts: four replies identical to the last byte, two diverging at a near-tie 158 and 447 characters in. Drafted against undrafted: identical on every prompt, on both profiles. The vision probe answers with the same reply hash as the MLX pair. Numbers: README "Measured".
