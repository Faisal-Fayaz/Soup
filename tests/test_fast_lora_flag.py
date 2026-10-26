"""#792 — ``training.fast_lora``: the switch that puts the Fast-LoRA kernels
(``utils/fast_lora.py``, ``fast_lora_mlp.py``, ``fast_lora_qkv.py``) on a run.

What is checked here runs on CPU in float32 and float64: the settings the flag
refuses, the wiring on the resident and the streamed SFT paths, and parity with
peft's own modules through the switch. The bf16 / fp16, NF4 and streamed-CUDA
rows of the issue are ``gpu`` tests at the end of the file.
"""

from __future__ import annotations

import json
import types

import pytest
from rich.console import Console

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

CONSOLE = Console()
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ON = types.SimpleNamespace(fast_lora=True)
OFF = types.SimpleNamespace(fast_lora=False)


def _cfg(task="sft", data=None, backend="transformers", modality="text", **training):
    from soup_cli.config.schema import SoupConfig

    lora = {"dropout": 0.0, **training.pop("lora", {})}
    return SoupConfig(
        base="test/model",
        task=task,
        backend=backend,
        modality=modality,
        data={"train": "tests/fixtures/sample_train.jsonl", **(data or {})},
        training={"fast_lora": True, "lora": lora, **training},
    )


def _refusal(**kwargs) -> str:
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        _cfg(**kwargs)
    return str(excinfo.value)


# ─── The flag and what it refuses ────────────────────────────────────────────


class TestTheFlag:
    def test_it_is_off_unless_asked_for(self):
        from soup_cli.config.schema import TrainingConfig

        assert TrainingConfig().fast_lora is False

    @pytest.mark.parametrize("quantization", ["none", "4bit"])
    def test_the_first_slice_loads(self, quantization):
        assert _cfg(quantization=quantization).training.fast_lora is True

    def test_the_default_lora_dropout_is_refused_with_the_way_out(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import SoupConfig

        with pytest.raises(ValidationError) as excinfo:
            SoupConfig(
                base="test/model",
                data={"train": "tests/fixtures/sample_train.jsonl"},
                training={"fast_lora": True},
            )
        message = str(excinfo.value)
        assert "training.fast_lora needs lora.dropout: 0" in message
        assert "Soup's default is 0.05" in message

    def test_another_backend_is_refused(self):
        assert "training.fast_lora needs backend='transformers'" in _refusal(backend="unsloth")

    def test_another_modality_is_refused(self):
        assert "training.fast_lora supports modality='text' only" in _refusal(
            modality="vision", data={"format": "llava"}
        )

    @pytest.mark.parametrize("task", ["dpo", "orpo", "kto"])
    def test_another_task_is_refused(self, task):
        data = {"format": "kto" if task == "kto" else "dpo"}
        assert "training.fast_lora supports task='sft' only" in _refusal(task=task, data=data)

    def test_no_lora_is_refused(self):
        assert "training.fast_lora needs a LoRA; lora.r=0" in _refusal(
            quantization="none", lora={"r": 0}
        )

    @pytest.mark.parametrize("adapter", ["use_dora", "use_vera"])
    def test_another_adapter_is_refused(self, adapter):
        assert f"training.fast_lora cannot be combined with lora.{adapter}" in _refusal(
            quantization="none", lora={adapter: True}
        )

    def test_target_parameters_are_refused(self):
        assert "cannot be combined with lora.target_parameters" in _refusal(
            quantization="none", lora={"target_parameters": ["mlp.experts.gate_up_proj"]}
        )

    @pytest.mark.parametrize("quantization", ["8bit", "gptq"])
    def test_another_quantization_is_refused(self, quantization):
        assert "training.fast_lora supports quantization 'none' or '4bit'" in _refusal(
            quantization=quantization
        )

    def test_use_liger_is_refused(self):
        assert "training.fast_lora cannot be combined with training.use_liger" in _refusal(
            use_liger=True
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [("unfrozen_parameters", ["model.layers.0.mlp"]), ("lisa_enabled", True)],
        ids=["spectrum", "lisa"],
    )
    def test_a_full_fine_tuning_mode_is_refused(self, field, value):
        """They train the model's own weights and skip the LoRA setup the
        kernels hook into, so the flag would be ignored. The default rank 64,
        as a user who sets only these would have it."""
        from pydantic import ValidationError

        from soup_cli.config.schema import SoupConfig

        kwargs = {
            "base": "test/model",
            "data": {"train": "tests/fixtures/sample_train.jsonl"},
        }
        training = {"quantization": "none", "lora": {"dropout": 0.0}, field: value}
        SoupConfig(**kwargs, training=training)  # the premise: it loads without the flag
        with pytest.raises(ValidationError) as excinfo:
            SoupConfig(**kwargs, training={**training, "fast_lora": True})
        assert f"training.fast_lora cannot be combined with training.{field}" in str(excinfo.value)

    def test_fp8_training_is_refused(self):
        assert (
            "training.fast_lora cannot be combined with training.quantization_aware='fp8'"
            in _refusal(quantization="none", quantization_aware="fp8")
        )


# ─── The switch on a model ───────────────────────────────────────────────────


def _model(seed=0, dtype=None, hidden=32):
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    lora = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none", target_modules=TARGETS)
    model = get_peft_model(LlamaForCausalLM(config), lora)
    with torch.no_grad():  # a trained adapter: lora_B at zero would hide a wrong path
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.1)
    return model.to(dtype) if dtype is not None else model


def _pair(**kwargs):
    """The same model twice; the second has the switch on."""
    from soup_cli.utils.peft_wiring import apply_fast_lora_patches

    plain, fast = _model(**kwargs), _model(**kwargs)
    counts = apply_fast_lora_patches(fast, ON)
    return plain, fast, counts


def _ids(seed=1):
    torch.manual_seed(seed)
    return torch.randint(0, 64, (2, 7))


def _lora_grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


class TestTheSwitch:
    def test_off_leaves_peft_alone(self):
        from soup_cli.utils.peft_wiring import apply_fast_lora_patches

        model = _model()
        assert apply_fast_lora_patches(model, OFF) is None
        assert not any("forward" in vars(m) for m in model.modules())

    def test_on_puts_every_kernel_on_its_modules(self):
        _, fast, counts = _pair()

        # Two layers: q/k/v and the MLP are each one group per layer, and o_proj
        # is the only projection left for the single-projection kernel.
        assert counts == {"q/k/v": 2, "MLP": 2, "single projection": 2}
        layer = fast.base_model.model.model.layers[0]
        x = torch.randn(1, 3, 32, requires_grad=True)
        assert "Fast" in type(layer.self_attn.o_proj(x).grad_fn).__name__
        assert "Fast" in type(layer.mlp(x).grad_fn).__name__

    def test_on_with_nothing_to_patch_is_refused(self):
        from soup_cli.utils.peft_wiring import apply_fast_lora_patches

        with pytest.raises(ValueError, match="no LoRA module was patched"):
            apply_fast_lora_patches(torch.nn.Sequential(torch.nn.Linear(4, 4)), ON)

    def test_a_conv1d_only_model_is_refused(self):
        """GPT-2's Conv1D LoRA (``fan_in_fan_out``) is patched but delegates to
        peft on every call: nothing the kernels can run, so the switch refuses
        and takes its patches off."""
        from peft import LoraConfig, get_peft_model
        from transformers import GPT2Config, GPT2LMHeadModel

        from soup_cli.utils.peft_wiring import apply_fast_lora_patches

        torch.manual_seed(0)
        config = GPT2Config(vocab_size=64, n_positions=32, n_embd=16, n_layer=2, n_head=2)
        lora = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["c_attn"])
        model = get_peft_model(GPT2LMHeadModel(config), lora)

        with pytest.raises(ValueError, match="no LoRA module was patched"):
            apply_fast_lora_patches(model, ON)
        assert not any("forward" in vars(m) for m in model.modules())

    def test_the_setup_line_names_each_kernel(self):
        from soup_cli.utils.peft_wiring import fast_lora_summary

        line = fast_lora_summary({"q/k/v": 2, "MLP": 2, "single projection": 2})
        assert line == "Fast-LoRA kernels on: 2 q/k/v, 2 MLP, 2 single projection"


class TestParityThroughTheSwitch:
    """fp32 against peft's own modules; forward and backward reported apart."""

    def test_forward(self):
        plain, fast, _ = _pair()
        ids = _ids()

        torch.testing.assert_close(fast(input_ids=ids).logits, plain(input_ids=ids).logits)

    def test_backward(self):
        plain, fast, _ = _pair()
        ids = _ids()
        for model in (plain, fast):
            model(input_ids=ids, labels=ids).loss.backward()

        plain_grads, fast_grads = _lora_grads(plain), _lora_grads(fast)
        assert plain_grads.keys() == fast_grads.keys() and len(plain_grads) == 28
        for name in plain_grads:
            torch.testing.assert_close(fast_grads[name], plain_grads[name], msg=name)

    def test_gradients_are_the_same_with_non_reentrant_checkpointing(self):
        _, fast, _ = _pair()
        _, checkpointed, _ = _pair()
        checkpointed.base_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        checkpointed.train()
        fast.train()
        ids = _ids()
        for model in (fast, checkpointed):
            model(input_ids=ids, labels=ids).loss.backward()

        for name, grad in _lora_grads(fast).items():
            torch.testing.assert_close(_lora_grads(checkpointed)[name], grad, msg=name)

    def test_with_the_adapter_disabled_the_output_is_the_base_models(self):
        plain, fast, _ = _pair()
        ids = _ids()
        with fast.disable_adapter():
            fast_base = fast(input_ids=ids).logits
        with plain.disable_adapter():
            plain_base = plain(input_ids=ids).logits

        torch.testing.assert_close(fast_base, plain_base)
        assert not torch.allclose(fast(input_ids=ids).logits, fast_base)  # the premise

    @pytest.mark.parametrize("path", ["o_proj", "mlp", "self_attn"])
    def test_gradcheck_in_float64(self, path):
        """Per kernel, w.r.t. its input and its LoRA A and B, with the base
        weight frozen. (Not through the whole model: Llama's RMSNorm runs in
        float32 whatever the model's dtype, so peft fails that check too.)"""
        from torch.func import functional_call

        _, fast, _ = _pair(dtype=torch.float64, hidden=16)
        decoder = fast.base_model.model.model
        layer = decoder.layers[0]
        module = layer.self_attn.o_proj if path == "o_proj" else getattr(layer, path)
        lora = {
            n: p.detach().clone().requires_grad_(True)
            for n, p in module.named_parameters()
            if "lora_" in n
        }
        names = list(lora)
        torch.manual_seed(2)
        x = torch.randn(1, 3, 16, dtype=torch.float64, requires_grad=True)
        kwargs = {}
        if path == "self_attn":
            positions = torch.arange(3).unsqueeze(0)
            kwargs = {
                "position_embeddings": decoder.rotary_emb(x, positions),
                "attention_mask": None,
            }

        def run(inputs, *params):
            out = functional_call(module, dict(zip(names, params)), (inputs,), kwargs)
            return out[0] if isinstance(out, tuple) else out

        assert len(names) >= 2
        assert torch.autograd.gradcheck(run, (x, *lora.values()), eps=1e-6, atol=1e-5)

    def test_fifty_steps_lose_the_same_to_three_decimals(self):
        """One batch, fifty AdamW steps at lr 1e-3. (At lr 1e-2 the two runs
        stay bit-identical for 33 steps and then part, as any two float32 runs
        whose last bits differ do under AdamW at that rate.)"""
        plain, fast, _ = _pair()
        torch.manual_seed(3)
        ids = torch.randint(0, 64, (2, 9))
        losses = {}
        for name, model in (("plain", plain), ("fast", fast)):
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=1e-3)
            losses[name] = []
            for _ in range(50):
                loss = model(input_ids=ids, labels=ids).loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses[name].append(loss.item())

        # Equal to 3 decimals: never 1e-3 apart (rounding each to 3 places would
        # call 4.1935 and 4.1934 different).
        gap = max(abs(f - p) for f, p in zip(losses["fast"], losses["plain"]))
        CONSOLE.print(json.dumps({"fifty_step_max_loss_gap": gap}))
        assert gap < 1e-3
        assert losses["plain"][-1] < losses["plain"][0] - 0.1  # it trained


# ─── Through the real SFT setup ──────────────────────────────────────────────


def _sft_rows(n=4):
    return [
        {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]
        }
    ] * n


def _resident_wrapper(tmp_path, monkeypatch, *, fast_lora):
    import yaml

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from tests._tiny_hf_models import tiny_model_dir

    monkeypatch.chdir(tmp_path)
    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": tiny_model_dir("llama"),
                "task": "sft",
                "backend": "transformers",
                "data": {"train": "unused.jsonl", "max_length": 64, "chat_template": "chatml"},
                "training": {
                    "epochs": 1,
                    "batch_size": 2,
                    "quantization": "none",
                    "fast_lora": fast_lora,
                    "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": TARGETS},
                },
                "output": str(tmp_path / "out"),
            }
        )
    )
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    wrapper.setup({"train": _sft_rows()})
    return wrapper


def _kernel_nodes(model):
    return {type(m).__name__ for m in model.modules() if "forward" in vars(m)}


class TestTheResidentSetup:
    def test_the_switch_reaches_the_model_the_trainer_trains(self, tmp_path, monkeypatch, capsys):
        wrapper = _resident_wrapper(tmp_path, monkeypatch, fast_lora=True)

        modules = dict(wrapper.trainer.model.named_modules())
        patched = [n for n, m in modules.items() if "forward" in vars(m)]
        assert any(
            hasattr(m, "_soup_fast_lora_qkv") for n, m in modules.items() if n.endswith("self_attn")
        )
        assert any(n.endswith("mlp") for n in patched)
        assert any(n.endswith("o_proj") for n in patched)
        assert "Fast-LoRA kernels on:" in capsys.readouterr().out

    def test_without_the_switch_nothing_is_patched(self, tmp_path, monkeypatch):
        wrapper = _resident_wrapper(tmp_path, monkeypatch, fast_lora=False)

        assert not any("forward" in vars(m) for m in wrapper.trainer.model.modules())


class TestAFailedPatchReleasesTheStreamedRuntime:
    """The switch runs after the runtime is built and before training's
    cleanup context starts: a failure there closes the runtime itself."""

    def _setup(self, tmp_path, monkeypatch):
        """The real streamed setup; returns, for each close, whether a runtime
        was there to close."""
        from soup_cli.trainer.stream_setup import StreamingSetupMixin
        from tests.test_v07204 import _build_streamed_wrapper

        closes = []
        real_close = StreamingSetupMixin._close_stream_runtime

        def counting_close(self):
            closes.append(getattr(self, "_stream_runtime", None) is not None)
            return real_close(self)

        monkeypatch.setattr(StreamingSetupMixin, "_close_stream_runtime", counting_close)
        with pytest.raises(Exception) as excinfo:
            _build_streamed_wrapper(
                tmp_path,
                monkeypatch,
                task="sft",
                n_layers=2,
                device="cpu",
                tie=False,
                fast_lora=True,
                lora={"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": TARGETS},
            )
        return closes, excinfo.value

    def test_nothing_patched(self, tmp_path, monkeypatch):
        from soup_cli.utils import fast_lora, fast_lora_mlp, fast_lora_qkv

        monkeypatch.setattr(fast_lora_qkv, "patch_fast_lora_qkv", lambda model: 0)
        monkeypatch.setattr(fast_lora_mlp, "patch_fast_lora_mlp", lambda model: 0)
        monkeypatch.setattr(fast_lora, "patch_fast_lora_single_projection", lambda model: 0)
        monkeypatch.setattr(fast_lora, "eligible_single_projections", lambda model: 0)

        closes, error = self._setup(tmp_path, monkeypatch)

        assert "no LoRA module was patched" in str(error)
        assert closes == [True]

    def test_a_patcher_that_raises(self, tmp_path, monkeypatch):
        from soup_cli.utils import fast_lora_qkv

        def boom(model):
            raise RuntimeError("patcher failed")

        monkeypatch.setattr(fast_lora_qkv, "patch_fast_lora_qkv", boom)

        closes, error = self._setup(tmp_path, monkeypatch)

        assert str(error) == "patcher failed"
        assert closes == [True]


class TestTheStreamedSetup:
    """A streamed tiny Llama with more layers than ``stream_buffers``: the LoRA
    gradients on every layer equal those of the same run without the switch."""

    def _grads(self, tmp_path, monkeypatch, fast_lora):
        from tests.test_v07204 import _build_streamed_wrapper

        wrapper, _, _ = _build_streamed_wrapper(
            tmp_path,
            monkeypatch,
            task="sft",
            n_layers=4,
            device="cpu",
            tie=False,
            stream_buffers=2,
            fast_lora=fast_lora,
            lora={"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": TARGETS},
        )
        try:
            with torch.no_grad():
                for name, param in wrapper.model.named_parameters():
                    if "lora_B" in name:
                        torch.manual_seed(abs(hash(name)) % 2**31)
                        param.normal_(std=0.1)
            ids = torch.tensor([[1, 5, 9, 13, 2, 7, 11]])
            out = wrapper.model(input_ids=ids, labels=ids)
            out.loss.backward()
            patched = sum("forward" in vars(m) for m in wrapper.model.modules())
            return _lora_grads(wrapper.model), patched
        finally:
            wrapper._close_stream_runtime()

    def test_every_layers_gradients_match_without_the_switch(self, tmp_path, monkeypatch):
        plain, plain_patched = self._grads(tmp_path / "plain", monkeypatch, False)
        fast, fast_patched = self._grads(tmp_path / "fast", monkeypatch, True)

        assert plain_patched < fast_patched  # the premise: the kernels are on
        layers = {name.split(".layers.")[1].split(".")[0] for name in fast if ".layers." in name}
        assert layers == {"0", "1", "2", "3"}
        assert plain.keys() == fast.keys()
        for name in plain:
            assert torch.count_nonzero(plain[name]) > 0, name
            torch.testing.assert_close(fast[name], plain[name], msg=name)


# ─── Rows that need CUDA (not run on CPU CI) ─────────────────────────────────


def _half_error(dtype):
    """Max-abs error of the fast and the peft path against float64, per output."""
    plain, fast, _ = _pair()
    reference = _model().double()
    ids = _ids().cuda()
    plain, fast, reference = plain.cuda().to(dtype), fast.cuda().to(dtype), reference.cuda()
    want = reference(input_ids=ids).logits
    plain_err = (plain(input_ids=ids).logits.double() - want).abs().max().item()
    fast_err = (fast(input_ids=ids).logits.double() - want).abs().max().item()
    return plain_err, fast_err


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_half_precision_forward_error_is_within_twice_peft(dtype):
    plain_err, fast_err = _half_error(getattr(torch, dtype))
    CONSOLE.print(json.dumps({"dtype": dtype, "peft_max_abs": plain_err, "fast_max_abs": fast_err}))
    assert fast_err <= 2 * plain_err


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_half_precision_backward_error_is_within_twice_peft(dtype):
    dtype = getattr(torch, dtype)
    plain, fast, _ = _pair()
    reference = _model().double().cuda()
    ids = _ids().cuda()
    reference(input_ids=ids, labels=ids).loss.backward()
    want = _lora_grads(reference)
    errors = {}
    for name, model in (("peft", plain), ("fast", fast)):
        model = model.cuda().to(dtype)
        model(input_ids=ids, labels=ids).loss.backward()
        got = _lora_grads(model)
        errors[name] = max((got[n].double() - want[n]).abs().max().item() for n in want)
    CONSOLE.print(json.dumps({"dtype": str(dtype), **errors}))
    assert errors["fast"] <= 2 * errors["peft"]


@pytest.mark.gpu
def test_nf4_matches_peft_over_the_same_linear4bit(tmp_path, monkeypatch):
    """Reference: an fp64 oracle built from the same NF4 weights (the
    dequantised ones, as in the #837 MLP oracle), with the #792 criterion
    fast_error <= 2 * peft_error + 1e-8 per output and per LoRA gradient.
    Every measured ratio is printed. (Bit-parity against peft never held on
    a real card: the fused kernels re-order reductions.)"""
    _run_resident_nf4_row(tmp_path, monkeypatch)


def _resident_wrapper_4bit(tmp_path, monkeypatch, *, fast_lora, quantization="4bit",
                           device="cuda", base_dir=None):
    """Resident SFT fixture; the historical name stays, but the #792 oracle
    rows also build a ``quantization="none"`` fp64 twin through this code
    path, on the streamed fixture's own tiny model dir so every shape and
    weight matches the arms."""
    import yaml

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from tests._tiny_hf_models import tiny_model_dir

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": str(base_dir) if base_dir else tiny_model_dir("llama"),
                "task": "sft",
                "data": {"train": "unused.jsonl", "max_length": 64, "chat_template": "chatml"},
                "training": {
                    "epochs": 1,
                    "batch_size": 2,
                    "quantization": quantization,
                    "fast_lora": fast_lora,
                    "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": TARGETS},
                },
                "output": str(tmp_path / "out"),
            }
        )
    )
    wrapper = SFTTrainerWrapper(cfg, device=device)
    wrapper.setup({"train": _sft_rows()})
    torch.manual_seed(4)
    with torch.no_grad():
        for name, param in wrapper.model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.1)
    return wrapper.model


@pytest.mark.gpu
@pytest.mark.parametrize("quantization", ["none", "4bit"])
def test_streamed_on_cuda_every_layer_matches(tmp_path, monkeypatch, quantization):
    """bf16 (emulated on sm_75) and NF4 on the GPU, streamed on both sides
    (#776), every layer among the checked grads; #792 (a): the fp64-oracle
    criterion replaces bit-parity against plain streaming. Prints every
    ratio."""
    _run_streamed_row(tmp_path, monkeypatch, quantization)


def _gpu_report(kind: str) -> dict:
    """Card, driver, SM clock and library versions, printed with the ratios
    so the numbers stay attributable to the card (#792 acceptance item 5)."""
    import subprocess

    import bitsandbytes as bnb
    import peft

    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    bf16_mode = f"{'emulated' if major < 8 else 'native'} bf16 ({arch})"
    payload = {
        "row": kind,
        "card": torch.cuda.get_device_name(0),
        "sm": arch,
        "bf16": bf16_mode,
        "torch": torch.__version__,
        "bitsandbytes": bnb.__version__,
        "peft": peft.__version__,
        "claim": "no end-to-end claim; parity readings on one card",
    }
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,clocks.sm",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=20,
        )
        payload["driver"], payload["sm_clock_mhz"] = out.stdout.strip().split(", ")
    except (OSError, subprocess.SubprocessError, ValueError):
        payload["driver"] = "nvidia-smi unavailable"
    return payload


def _apply_criterion(ratios, name, fast_value, peft_value, oracle):
    """#792 (a): fast_error <= 2 * peft_error + 1e-8 against the fp64 oracle.

    Mutates ``ratios[name]`` with the measured ratio, so a passing run still
    reports every ratio; returns (fast_error, peft_error) for the log."""
    fast_error = (fast_value.detach().double() - oracle).abs().max().item()
    peft_error = (peft_value.detach().double() - oracle).abs().max().item()
    ratio = fast_error / peft_error if peft_error else (float("inf") if fast_error else 0.0)
    ratios[name] = round(ratio, 6)
    assert fast_error <= 2.0 * peft_error + 1e-8, (
        f"{name}: fast error {fast_error:.6g} exceeds twice peft error "
        f"{peft_error:.6g} against the fp64 oracle"
    )
    return fast_error, peft_error


def _logits_and_grads(model, ids):
    """One forward+backward: logits captured before autograd frees them."""
    result = model(input_ids=ids, labels=ids)
    result.loss.backward()
    return result.logits.detach(), _lora_grads(model)


class _Fp64LoraProjection(torch.nn.Module):
    """fp64 stand-in for a peft ``lora.bnb.Linear4bit`` wrapper.

    Keeps the ``base_layer`` / ``lora_A`` / ``lora_B`` attribute names, so
    parameter names match the arms for grad-by-name comparison. The base is
    the dequantised NF4 weight widened to float64 — the same weights the
    arms compute with, minus their low precision, as in the #837 oracle."""

    def __init__(self, base, wrapper, device):
        super().__init__()
        from bitsandbytes.functional import dequantize_4bit

        dense = dequantize_4bit(base.weight.data, base.quant_state).to(torch.float64)
        self.base_layer = torch.nn.Linear(
            dense.shape[1], dense.shape[0], bias=False,
            dtype=torch.float64, device=device,
        )
        self.base_layer.weight.data.copy_(dense)
        self.lora_A = torch.nn.ModuleDict()
        self.lora_B = torch.nn.ModuleDict()
        adapter = "default"
        lora_a = wrapper.lora_A[adapter].weight.detach().to(torch.float64)
        self.lora_A[adapter] = torch.nn.Linear(
            lora_a.shape[1], lora_a.shape[0], bias=False,
            dtype=torch.float64, device=device,
        )
        self.lora_A[adapter].weight.data.copy_(lora_a)
        lora_b = wrapper.lora_B[adapter].weight.detach().to(torch.float64)
        self.lora_B[adapter] = torch.nn.Linear(
            lora_b.shape[1], lora_b.shape[0], bias=False,
            dtype=torch.float64, device=device,
        )
        self.lora_B[adapter].weight.data.copy_(lora_b)
        self.adapter = adapter
        self._scaling = float(wrapper.scaling[adapter])

    def forward(self, x):
        lora = self.lora_B[self.adapter](self.lora_A[self.adapter](x))
        return self.base_layer(x) + lora * self._scaling


def _fp64_twin(model):
    """Return an fp64 twin of a resident NF4 model.

    Each adapted NF4 wrapper is replaced with :class:`_Fp64LoraProjection`
    (dequantised base, adapters widened); everything else is widened with
    ``double()`` in place. The base problem — weights and seeded adapters —
    stays the one the arms run."""
    import copy

    from peft.tuners.lora.bnb import Linear4bit as LoraLinear4bit

    twin = copy.deepcopy(model)
    for path, module in list(twin.named_modules()):
        if isinstance(module, LoraLinear4bit):
            replacement = _Fp64LoraProjection(
                module.base_layer, module,
                device=module.base_layer.weight.device,
            )
            parent, _, leaf = path.rpartition(".")
            setattr(twin.get_submodule(parent), leaf, replacement)
    return twin.double()


def _oracle_lora_grads(oracle):
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in oracle.named_parameters() if "lora_" in name
    }


def _run_criterion(kind, peft, fast, oracle_logits, oracle_grads):
    """Assert the #792 criterion on logits (one output) and every adapter
    gradient; print every ratio (partial report on violation, so a failing
    run still hands over the ratios measured so far, per #792 item 4)."""
    ratios = {}
    errors = {}
    errors["logits"] = _apply_criterion(
        ratios, "logits", fast["logits"], peft["logits"], oracle_logits,
    )
    for pname, grad in oracle_grads.items():
        try:
            errors[pname] = _apply_criterion(
                ratios, pname, fast["grads"][pname], peft["grads"][pname], grad,
            )
        except AssertionError:
            errors[pname] = None
            # print() rather than Rich: pytest captures stdout cleanly with
            # `-s`, and the intact single-line JSON matches #837's style.
            print("#792 partial " + json.dumps(
                {**_gpu_report(kind), "ratios": ratios, "errors": errors}
            ))
            raise
    print("#792 " + json.dumps(
        {**_gpu_report(kind), "ratios": ratios, "errors": errors}
    ))


def _run_resident_nf4_row(tmp_path, monkeypatch):
    plain = _resident_wrapper_4bit(tmp_path / "plain", monkeypatch, fast_lora=False)
    fast = _resident_wrapper_4bit(tmp_path / "fast", monkeypatch, fast_lora=True)
    oracle = _fp64_twin(plain)
    ids = torch.tensor([[1, 5, 9, 13, 2, 7, 11]], device="cuda")

    oracle_out = oracle(input_ids=ids, labels=ids)
    oracle_out.loss.backward()
    arms = {
        name: {
            "logits": logits, "grads": grads,
        } for name, (logits, grads) in {
            "peft": _logits_and_grads(plain, ids),
            "fast": _logits_and_grads(fast, ids),
        }.items()
    }
    _run_criterion(
        "resident-nf4", arms["peft"], arms["fast"],
        oracle_out.logits, _oracle_lora_grads(oracle),
    )


def _run_streamed_row(tmp_path, monkeypatch, quantization):
    from tests.test_v07204 import _build_streamed_wrapper

    ids = torch.tensor([[1, 5, 9, 13, 2, 7, 11]], device="cuda")
    arms = {}
    plain_weights = None
    import os
    for name, flag in (("peft", False), ("fast", True)):
        wrapper, _, weights = _build_streamed_wrapper(
            tmp_path / name,
            monkeypatch,
            task="sft",
            n_layers=4,
            device="cuda",
            tie=False,
            stream_buffers=2,
            quantization=quantization,
            fast_lora=flag,
            lora={"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": TARGETS},
        )
        if name == "peft":
            plain_weights = os.path.abspath(weights)
        try:
            torch.manual_seed(5)
            with torch.no_grad():
                for pname, param in wrapper.model.named_parameters():
                    if "lora_B" in pname:
                        param.normal_(std=0.1)
            logits, grads = _logits_and_grads(wrapper.model, ids)
            arms[name] = {
                "logits": logits, "grads": grads,
                "params": {
                    pname: param.detach().clone()
                    for pname, param in wrapper.model.named_parameters()
                    if "lora_" in pname
                },
            }
        finally:
            wrapper._close_stream_runtime()

    # fp64 oracle from the same weights, on the streamed fixture's own tiny
    # model dir so shapes and weights match the arms exactly: resident NF4 ->
    # dequantised twin; bf16 streaming -> the float32 twin widened exactly.
    if quantization == "4bit":
        oracle = _fp64_twin(_resident_wrapper_4bit(
            tmp_path / "oracle", monkeypatch, fast_lora=False, base_dir=plain_weights,
        ))
    else:
        oracle = _resident_wrapper_4bit(
            tmp_path / "oracle", monkeypatch, fast_lora=False,
            quantization="none", base_dir=plain_weights,
        ).double()
    # The streamed arms draw their lora_B themselves (seed 5 per arm); the
    # oracle inherits the exact same problem by copying those parameters.
    with torch.no_grad():
        for name, parameter in oracle.named_parameters():
            if name in arms["peft"]["params"]:
                parameter.copy_(arms["peft"]["params"][name].to(parameter.dtype))
    oracle_out = oracle(input_ids=ids, labels=ids)
    oracle_out.loss.backward()

    _run_criterion(
        f"streamed-{quantization}", arms["peft"], arms["fast"],
        oracle_out.logits, _oracle_lora_grads(oracle),
    )


@pytest.mark.gpu
def test_rows_fail_when_a_kernel_drops_the_scaling(tmp_path, monkeypatch):
    """#792 guard: the 2x rule is loose when the peft error is large, so the
    rewritten rows must still fail on a wrong kernel. A deliberate kernel
    break — dropping the scaling factor at the shared saturated-add seam all
    three kernels call — must make every row raise; otherwise the criterion
    would silently pass wrong kernels."""
    import soup_cli.utils.fast_lora as fast_lora_single
    import soup_cli.utils.fast_lora_mlp as fast_lora_mlp
    import soup_cli.utils.fast_lora_qkv as fast_lora_qkv

    real_add = fast_lora_single._scaled_lora_add

    def unscaled_add(out, hidden, lora_b, scaling):
        return real_add(out, hidden, lora_b, 1.0)

    monkeypatch.setattr(fast_lora_single, "_scaled_lora_add", unscaled_add)
    monkeypatch.setattr(fast_lora_mlp, "_scaled_lora_add", unscaled_add)
    monkeypatch.setattr(fast_lora_qkv, "_scaled_lora_add", unscaled_add)

    with pytest.raises(AssertionError):
        _run_resident_nf4_row(tmp_path, monkeypatch)
    for quantization in ("none", "4bit"):
        with pytest.raises(AssertionError):
            _run_streamed_row(tmp_path, monkeypatch, quantization)
