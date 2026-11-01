"""GPT-2 adapter (Hugging Face `GPT2LMHeadModel`).

Real module layout, verified against transformers 5.20.1 on the meta
device:

    transformer.wte   token embedding      (vocab, n_embd)
    transformer.wpe   position embedding   (n_positions, n_embd)
    transformer.drop  dropout (no parameters)
    transformer.h[i]  block: ln_1, attn.c_attn (n_embd, 3*n_embd),
                      attn.c_proj, ln_2, mlp.c_fc (n_embd, inner),
                      mlp.c_proj (inner, n_embd)
    transformer.ln_f  final LayerNorm
    lm_head           (vocab, n_embd), TIED to transformer.wte.weight

GPT-2 uses Conv1D, so linear weights are stored (in_features,
out_features). Partition units, in order:
    embeddings (wte + wpe + drop), block.0 .. block.{n_layer-1},
    final_norm (ln_f), lm_head
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from partitioner import model_spec as ms
from partitioner.adapters.base import AdapterError, ArchitectureAdapter


class GPT2Adapter(ArchitectureAdapter):
    name = "gpt2"
    version = 1
    model_types = ("gpt2",)

    def validate_config(self, config: Any) -> None:
        problems: List[str] = []
        for field in ("n_layer", "n_embd", "n_head", "vocab_size", "n_positions"):
            value = getattr(config, field, None)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                problems.append(f"{field} must be a positive integer, got {value!r}")
        n_embd, n_head = getattr(config, "n_embd", None), getattr(config, "n_head", None)
        if isinstance(n_embd, int) and isinstance(n_head, int) and n_head > 0 and n_embd % n_head:
            problems.append(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        inner = getattr(config, "n_inner", None)
        if inner is not None and (not isinstance(inner, int) or inner <= 0):
            problems.append(f"n_inner must be a positive integer or null, got {inner!r}")
        if getattr(config, "add_cross_attention", False):
            problems.append("cross-attention GPT-2 variants are not supported")
        if problems:
            raise AdapterError("invalid GPT-2 configuration: " + "; ".join(problems))

    def build_meta_model(self, config: Any):
        import torch
        from transformers import GPT2LMHeadModel

        with torch.device("meta"):
            return GPT2LMHeadModel(config)

    def describe(self, model_id: str, config: Any, model: Any) -> ms.ModelSpec:
        self.validate_config(config)
        n_layer, h = config.n_layer, config.n_embd
        inner = config.n_inner or 4 * h
        vocab, n_pos = config.vocab_size, config.n_positions

        # Identify tied parameters by tensor identity -- not by trusting
        # the config flag alone.
        owner_by_id: Dict[int, str] = {}
        tied: List[Tuple[str, str]] = []
        params: Dict[str, Any] = {}
        for name, p in model.named_parameters(remove_duplicate=False):
            if p.device.type != "meta":
                raise AdapterError(f"parameter {name} is on {p.device}, expected the meta device (no real weights)")
            if id(p) in owner_by_id:
                tied.append((name, owner_by_id[id(p)]))
            else:
                owner_by_id[id(p)] = name
            params[name] = p
        tied_map = dict(tied)

        def info(name: str, kind: str) -> ms.ParamInfo:
            if name not in params:
                raise AdapterError(f"expected parameter {name} not found in the model")
            p = params[name]
            return ms.ParamInfo(name, tuple(p.shape), p.numel(), kind, tied_map.get(name))

        def check_shape(name: str, expected: Tuple[int, ...]) -> None:
            actual = tuple(params[name].shape)
            if actual != expected:
                raise AdapterError(f"{name} has shape {actual}, expected {expected} from the config")

        comps: List[ms.Component] = []
        units: List[ms.Unit] = []

        check_shape("transformer.wte.weight", (vocab, h))
        check_shape("transformer.wpe.weight", (n_pos, h))
        comps.append(ms.Component("transformer.wte", ms.EMBEDDING, (info("transformer.wte.weight", ms.EMBEDDING_TABLE),)))
        comps.append(ms.Component("transformer.wpe", ms.POSITIONAL_EMBEDDING, (info("transformer.wpe.weight", ms.EMBEDDING_TABLE),)))
        comps.append(ms.Component("transformer.drop", ms.PARAMETERLESS, ()))
        units.append(ms.Unit(0, "embeddings", ms.EMBEDDING, ("transformer.wte", "transformer.wpe", "transformer.drop")))

        if len(model.transformer.h) != n_layer:
            raise AdapterError(f"model has {len(model.transformer.h)} blocks but config.n_layer = {n_layer}")
        block_shapes = {
            "ln_1.weight": ((h,), ms.NORM), "ln_1.bias": ((h,), ms.NORM),
            "attn.c_attn.weight": ((h, 3 * h), ms.LINEAR_WEIGHT), "attn.c_attn.bias": ((3 * h,), ms.BIAS),
            "attn.c_proj.weight": ((h, h), ms.LINEAR_WEIGHT), "attn.c_proj.bias": ((h,), ms.BIAS),
            "ln_2.weight": ((h,), ms.NORM), "ln_2.bias": ((h,), ms.NORM),
            "mlp.c_fc.weight": ((h, inner), ms.LINEAR_WEIGHT), "mlp.c_fc.bias": ((inner,), ms.BIAS),
            "mlp.c_proj.weight": ((inner, h), ms.LINEAR_WEIGHT), "mlp.c_proj.bias": ((h,), ms.BIAS),
        }
        for i in range(n_layer):
            prefix = f"transformer.h.{i}."
            actual = {n[len(prefix):] for n in params if n.startswith(prefix)}
            if actual != set(block_shapes):
                raise AdapterError(
                    f"block {i} parameters differ from the GPT-2 layout: "
                    f"unexpected {sorted(actual - set(block_shapes))}, missing {sorted(set(block_shapes) - actual)}"
                )
            block_params = []
            for suffix, (shape, kind) in block_shapes.items():
                check_shape(prefix + suffix, shape)
                block_params.append(info(prefix + suffix, kind))
            comps.append(ms.Component(f"transformer.h.{i}", ms.TRANSFORMER_BLOCK, tuple(block_params), block_index=i))
            units.append(ms.Unit(i + 1, f"block.{i}", ms.TRANSFORMER_BLOCK, (f"transformer.h.{i}",), block_index=i))

        check_shape("transformer.ln_f.weight", (h,))
        comps.append(ms.Component("transformer.ln_f", ms.FINAL_NORM,
                                  (info("transformer.ln_f.weight", ms.NORM), info("transformer.ln_f.bias", ms.NORM))))
        units.append(ms.Unit(n_layer + 1, "final_norm", ms.FINAL_NORM, ("transformer.ln_f",)))

        check_shape("lm_head.weight", (vocab, h))
        comps.append(ms.Component("lm_head", ms.LM_HEAD, (info("lm_head.weight", ms.LINEAR_WEIGHT),)))
        units.append(ms.Unit(n_layer + 2, "lm_head", ms.LM_HEAD, ("lm_head",)))

        # Strictness: every module with parameters, and every direct
        # child of the backbone, must be accounted for.
        assigned = {p.name for c in comps for p in c.params}
        unassigned = sorted(set(params) - assigned)
        if unassigned:
            raise AdapterError(f"parameters not mapped to any component: {unassigned}")
        known_children = {"wte", "wpe", "drop", "h", "ln_f"}
        children = {n for n, _ in model.transformer.named_children()}
        if children != known_children:
            raise AdapterError(f"unexpected GPT-2 backbone modules: {sorted(children ^ known_children)}")

        total = sum(p.numel() for p in model.parameters())  # parameters() de-duplicates tied tensors
        return ms.ModelSpec(
            model_id=model_id,
            architecture=config.model_type,
            adapter=self.name,
            adapter_version=self.version,
            config={
                "model_type": config.model_type,
                "n_layer": n_layer,
                "n_embd": h,
                "n_head": config.n_head,
                "n_inner": config.n_inner,
                "vocab_size": vocab,
                "n_positions": n_pos,
                "activation_function": getattr(config, "activation_function", None),
                "layer_norm_epsilon": getattr(config, "layer_norm_epsilon", None),
                "tie_word_embeddings": getattr(config, "tie_word_embeddings", None),
            },
            config_fingerprint=self.fingerprint(config),
            n_layers=n_layer,
            hidden_size=h,
            intermediate_size=inner,
            vocab_size=vocab,
            max_positions=n_pos,
            attention=ms.AttentionShape(n_heads=config.n_head, n_kv_heads=config.n_head, head_dim=h // config.n_head),
            total_params=total,
            components=tuple(comps),
            units=tuple(units),
            tied_weights=tuple(tied),
        )
