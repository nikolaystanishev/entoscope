# Models and caching

Operations accept any of three model types:

- a `TransformerBridge`
- a `HookedTransformer`
- either of them wrapped in a `CachedModel`

## TransformerBridge

`TransformerBridge` is how transformer-lens 3 and later load a HuggingFace model. entoscope reads it as
it is, so its activations and weights are HuggingFace's own, and a run means the same as the same
computation done on the HuggingFace model directly:

```python
from transformer_lens.model_bridge import TransformerBridge

model = TransformerBridge.boot_transformers("gpt2")
model.eval()
```

`boot_transformers` hands back a model whose HuggingFace weights are already in eval mode, so the
`eval()` is a safeguard. It matters for a HuggingFace model you build yourself and pass in as
`hf_model=`: the bridge's `eval()` does not reach the model it wraps, so call `eval()` on that model
too, or its dropout stays on.

### Compatibility mode

`model.enable_compatibility_mode()` switches a bridge to `transformer_lens`'s own conventions. It
folds each LayerNorm into the weights that follow it, and centers the weights that write to the
residual stream. Both modes work with every operation. What changes is what the intermediate numbers
mean, not whether a run works.

One consequence shows in the parameter lens. GPT-2 ties its embedding and unembedding, so with
HuggingFace's weights `W_E.T` equals `W_U` and the lens's two projection spaces agree exactly; folding
and centering separate them.

## HookedTransformer

`HookedTransformer` is accepted wherever a bridge is, with the transformer-lens releases that have it:
2.x and 3.x, since transformer-lens 4 removed it in favour of the bridge. It is the only option on
Intel macOS, whose transformer-lens 2.15 has no bridge:

<!-- docs-test: skip -->
```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")
model.eval()
```

`from_pretrained` folds LayerNorm and centers weights by default, so its numbers match a bridge in
compatibility mode rather than a plain bridge.

## Caching activations on disk

`CachedModel` wraps a model and stores what `run_with_cache` returns on disk, so a forward pass a run
has already made is read back instead of recomputed. Everything else is forwarded to the wrapped
model, so an operation cannot tell the difference:

```python
from entoscope import CachedModel

cached = CachedModel(TransformerBridge.boot_transformers("gpt2"), cache_dir=".activation_cache")
cached.eval()
```

With `cache_dir=None` the wrapper passes every call straight through, which lets a script make
caching optional with a single argument. A method that returns the model itself, such as `to` or
`eval`, returns the wrapper, so `cached = cached.to("cuda")` keeps the cache, and the wrapper is called
as the model is.

### What gets cached

Only `run_with_cache` calls are cached, which are the runs an input makes before the method starts:

- the lens's prompt
- every probing example
- patching's source and target runs
- every circuit-discovery prompt, with the residual stream, MLP output, values and attention pattern
  of every layer

Patched runs and gradient passes go through `run_with_hooks` and are never cached. Circuit discovery
caches those activations of every layer per prompt, which takes far more disk than an index-filtered
run.

### What an entry is keyed by

An entry is stored at `<cache_dir>/<model name>/<digest>.pt`, written atomically. Its digest covers:

- the tokens, the requested activation names in any order, and every other `run_with_cache` argument
  that changes the result;
- a fingerprint of the model: its class, its config's dimensions and dtype, and the name, shape,
  dtype and first 64 values of every parameter;
- the version of the entry format, so an entry written by a release that stored it differently is
  recomputed rather than misread.

The weights in the fingerprint keep a fine-tune, a supplied `hf_model` or another revision from sharing
entries with the base model. A bridge and a `HookedTransformer` of the same model never share entries
either.

The fingerprint is taken on every call, so a change made after wrapping, such as
`enable_compatibility_mode()`, a change of dtype or a training step, is a new entry rather than a stale
one. Patching runs each prompt as wide as its patched runs, and a tensor whose rows come out alike bit
for bit is stored as one row. A backend that rounds the copies apart, as some x86 CPUs do, has every
row kept, so an entry always reads back exactly what the model computed. Entries are never evicted; delete the directory to clear the cache.
