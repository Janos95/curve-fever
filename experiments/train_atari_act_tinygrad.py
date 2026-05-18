import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

DEFAULT_WEBGPU_PATH = Path(
  "/Users/janosmeny/Projects/dawn-7069/out/Release/src/dawn/native/libwebgpu_dawn_tinygrad.dylib"
)

if DEFAULT_WEBGPU_PATH.exists():
  os.environ.setdefault("WEBGPU_PATH", str(DEFAULT_WEBGPU_PATH))
os.environ.setdefault("WEBGPU_BACKEND", "WGPUBackendType_Metal")

from tinygrad import Tensor, TinyJit, dtypes, nn  # noqa: E402
from tinygrad.nn.optim import AdamW  # noqa: E402
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save  # noqa: E402


class AtariChunkDataset:
  def __init__(self, root: Path, split: str):
    self.x = np.load(root / f"{split}_x.npy", mmap_mode="r")
    self.y = np.load(root / f"{split}_y.npy", mmap_mode="r")

  def __len__(self) -> int:
    return int(self.y.shape[0])

  def batch(self, indices: np.ndarray, device: str) -> tuple[Tensor, Tensor]:
    x = np.asarray(self.x[indices], dtype=np.uint8).astype(np.float32) / 255.0
    y = np.asarray(self.y[indices], dtype=np.int32)
    return Tensor(x, device=device).realize(), Tensor(y, dtype=dtypes.int32, device=device).realize()


class MultiHeadSelfAttention:
  def __init__(self, d_model: int, nhead: int, dropout: float):
    assert d_model % nhead == 0
    self.d_model = d_model
    self.nhead = nhead
    self.head_dim = d_model // nhead
    self.dropout = dropout
    self.qkv = nn.Linear(d_model, 3 * d_model)
    self.proj = nn.Linear(d_model, d_model)

  def __call__(self, x: Tensor) -> Tensor:
    batch, tokens, _ = x.shape
    qkv = self.qkv(x).reshape(batch, tokens, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
    query, key, value = qkv[0], qkv[1], qkv[2]
    attn = query.scaled_dot_product_attention(
      key,
      value,
      dropout_p=self.dropout if Tensor.training else 0.0,
    )
    attn = attn.permute(0, 2, 1, 3).reshape(batch, tokens, self.d_model)
    return self.proj(attn)


class TransformerBlock:
  def __init__(self, d_model: int, nhead: int, dropout: float):
    self.norm1 = nn.LayerNorm(d_model)
    self.attn = MultiHeadSelfAttention(d_model, nhead, dropout)
    self.norm2 = nn.LayerNorm(d_model)
    self.ff1 = nn.Linear(d_model, 4 * d_model)
    self.ff2 = nn.Linear(4 * d_model, d_model)
    self.dropout = dropout

  def __call__(self, x: Tensor) -> Tensor:
    x = x + self.attn(self.norm1(x)).dropout(self.dropout)
    x = x + self.ff2(self.ff1(self.norm2(x)).gelu().dropout(self.dropout)).dropout(self.dropout)
    return x


class AtariACT:
  def __init__(
    self,
    frame_stack: int = 4,
    image_size: int = 84,
    action_vocab: int = 18,
    chunk_len: int = 8,
    d_model: int = 256,
    nhead: int = 8,
    layers: int = 4,
    dropout: float = 0.1,
  ):
    self.chunk_len = chunk_len
    self.action_vocab = action_vocab
    self.d_model = d_model
    self.c1 = nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)
    self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
    self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
    flat = self._cnn(Tensor.zeros(1, frame_stack, image_size, image_size)).shape[-1]
    self.state_proj = nn.Linear(flat, d_model)
    self.state_ln = nn.LayerNorm(d_model)
    self.state_type = Tensor.zeros(1, 1, d_model, requires_grad=True)
    self.query = Tensor.randn(1, chunk_len, d_model, requires_grad=True) * 0.02
    self.pos = Tensor.randn(1, chunk_len + 1, d_model, requires_grad=True) * 0.02
    self.blocks = [TransformerBlock(d_model, nhead, dropout) for _ in range(layers)]
    self.head = nn.Linear(d_model, action_vocab)

  def _cnn(self, x: Tensor) -> Tensor:
    x = self.c1(x).relu()
    x = self.c2(x).relu()
    x = self.c3(x).relu()
    return x.flatten(1)

  def __call__(self, x: Tensor) -> Tensor:
    batch = x.shape[0]
    state = self.state_ln(self.state_proj(self._cnn(x))).reshape(batch, 1, self.d_model) + self.state_type
    queries = self.query.expand(batch, self.chunk_len, self.d_model)
    tokens = state.cat(queries, dim=1) + self.pos
    for block in self.blocks:
      tokens = block(tokens)
    return self.head(tokens[:, 1:])


def read_meta(data: Path) -> dict[str, int]:
  meta = {}
  for line in (data / "metadata.txt").read_text().splitlines():
    key, value = line.split("=", 1)
    meta[key] = int(value)
  return meta


def move_model(model: AtariACT, device: str) -> None:
  Tensor.realize(*[tensor.to_(device) for tensor in get_state_dict(model).values()])


def count_params(model: AtariACT) -> int:
  return sum(int(np.prod(p.shape)) for p in get_parameters(model))


def training_loss(model: AtariACT, x: Tensor, y: Tensor, action_vocab: int) -> Tensor:
  logits = model(x)
  return logits.reshape(-1, action_vocab).sparse_categorical_crossentropy(y.reshape(-1))


def eval_outputs(model: AtariACT, x: Tensor, y: Tensor, action_vocab: int) -> tuple[Tensor, Tensor, Tensor]:
  logits = model(x)
  loss = logits.reshape(-1, action_vocab).sparse_categorical_crossentropy(y.reshape(-1), reduction="sum")
  preds = logits.argmax(axis=-1)
  _, top3_preds = logits.topk(k=min(3, action_vocab), dim=-1)
  return loss, preds, top3_preds


def make_train_step(model: AtariACT, opt: AdamW, action_vocab: int, use_jit: bool):
  def step(x: Tensor, y: Tensor) -> Tensor:
    loss = training_loss(model, x, y, action_vocab)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss

  return TinyJit(step) if use_jit else step


def make_eval_step(model: AtariACT, action_vocab: int, use_jit: bool):
  def step(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return eval_outputs(model, x, y, action_vocab)

  return TinyJit(step) if use_jit else step


def batch_indices(total: int, batch_size: int, shuffle: bool, rng: np.random.Generator, max_batches: int | None = None):
  indices = np.arange(total)
  if shuffle:
    rng.shuffle(indices)
  for batch, start in enumerate(range(0, total, batch_size), 1):
    if max_batches is not None and batch > max_batches:
      break
    yield indices[start : min(start + batch_size, total)]


def evaluate(
  model: AtariACT,
  dataset: AtariChunkDataset,
  batch_size: int,
  device: str,
  max_batches: int | None = None,
  use_jit: bool = False,
) -> dict:
  Tensor.training = False
  eval_step = make_eval_step(model, model.action_vocab, use_jit)
  eager_eval_step = make_eval_step(model, model.action_vocab, False)
  total_loss = 0.0
  total_tokens = 0
  total_sequences = 0
  correct = 0
  first_correct = 0
  top3 = 0
  chunk_all = 0
  pos_correct = np.zeros(model.chunk_len, dtype=np.int64)
  target_counts = np.zeros(model.action_vocab, dtype=np.int64)
  pred_counts = np.zeros(model.action_vocab, dtype=np.int64)

  for idx in batch_indices(len(dataset), batch_size, False, np.random.default_rng(0), max_batches):
    x, y = dataset.batch(idx, device)
    step = eval_step if use_jit and len(idx) == batch_size else eager_eval_step
    loss, preds, top3_preds = step(x, y)
    Tensor.realize(loss, preds, top3_preds)

    y_np = y.numpy()
    pred_np = preds.numpy()
    top3_np = top3_preds.numpy()
    eq = pred_np == y_np

    total_loss += float(loss.numpy())
    total_tokens += int(y_np.size)
    total_sequences += int(y_np.shape[0])
    correct += int(eq.sum())
    first_correct += int(eq[:, 0].sum())
    top3 += int((top3_np == y_np[..., None]).any(axis=-1).sum())
    chunk_all += int(eq.all(axis=1).sum())
    pos_correct += eq.sum(axis=0)
    target_counts += np.bincount(y_np.reshape(-1), minlength=model.action_vocab)
    pred_counts += np.bincount(pred_np.reshape(-1), minlength=model.action_vocab)

  loss = total_loss / total_tokens
  return {
    "loss": loss,
    "perplexity": float(np.exp(loss)),
    "token_acc": correct / total_tokens,
    "first_step_acc": first_correct / total_sequences,
    "top3_acc": top3 / total_tokens,
    "full_chunk_acc": chunk_all / total_sequences,
    "pos_acc": (pos_correct / total_sequences).tolist(),
    "target_action_counts": target_counts.tolist(),
    "pred_action_counts": pred_counts.tolist(),
  }


def train(args: argparse.Namespace) -> None:
  Tensor.manual_seed(args.seed)
  rng = np.random.default_rng(args.seed)
  args.out.mkdir(parents=True, exist_ok=True)

  meta = read_meta(args.data)
  train_ds = AtariChunkDataset(args.data, "train")
  test_ds = AtariChunkDataset(args.data, "test")
  model = AtariACT(
    frame_stack=meta["frame_stack"],
    image_size=meta["image_size"],
    action_vocab=meta["action_vocab"],
    chunk_len=meta["chunk_len"],
    d_model=args.d_model,
    nhead=args.heads,
    layers=args.layers,
    dropout=args.dropout,
  )
  move_model(model, args.device)
  opt = AdamW(get_parameters(model), lr=args.lr, weight_decay=args.weight_decay, fused=args.fused_optim)
  train_step = make_train_step(model, opt, meta["action_vocab"], args.jit_train_step)
  eager_train_step = make_train_step(model, opt, meta["action_vocab"], False)

  print(f"device={args.device} webgpu_path={os.environ.get('WEBGPU_PATH', '<unset>')}")
  print(f"jit_train_step={args.jit_train_step} jit_eval_step={args.jit_eval_step} fused_optim={args.fused_optim}")
  print(f"params={count_params(model):,}")
  print(f"train={len(train_ds):,} test={len(test_ds):,}")
  metrics = {
    "device": args.device,
    "params": count_params(model),
    "train_samples": len(train_ds),
    "test_samples": len(test_ds),
    "meta": meta,
    "epochs": [],
  }

  for epoch in range(1, args.epochs + 1):
    Tensor.training = True
    start = time.perf_counter()
    running_loss = 0.0
    running_tokens = 0

    for step, idx in enumerate(batch_indices(len(train_ds), args.batch_size, True, rng, args.max_train_batches), 1):
      x, y = train_ds.batch(idx, args.device)
      step_fn = train_step if args.jit_train_step and len(idx) == args.batch_size else eager_train_step
      loss = step_fn(x, y)
      running_loss += float(loss.numpy()) * int(y.shape[0]) * meta["chunk_len"]
      running_tokens += int(y.shape[0]) * meta["chunk_len"]
      if step % args.log_interval == 0 or step == (len(train_ds) + args.batch_size - 1) // args.batch_size:
        print(f"epoch={epoch:02d} step={step:04d} loss={running_loss / running_tokens:.4f}", flush=True)

    eval_metrics = evaluate(model, test_ds, args.eval_batch_size, args.device, args.max_eval_batches, args.jit_eval_step)
    epoch_metrics = {
      "epoch": epoch,
      "train_loss": running_loss / running_tokens,
      "seconds": time.perf_counter() - start,
      "test": eval_metrics,
    }
    metrics["epochs"].append(epoch_metrics)
    print(json.dumps(epoch_metrics, indent=2), flush=True)
    with (args.out / "metrics.json").open("w") as f:
      json.dump(metrics, f, indent=2)

  if args.save:
    state = {name: tensor.detach().to("CPU").realize() for name, tensor in get_state_dict(model).items()}
    safe_save(state, str(args.save))
    print(f"saved={args.save}")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--data", type=Path, default=Path("../WorldModel/data/atari-act-stratified"))
  parser.add_argument("--out", type=Path, default=Path("runs/atari-act-tinygrad-webgpu"))
  parser.add_argument("--device", default="WEBGPU")
  parser.add_argument("--epochs", type=int, default=3)
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--eval-batch-size", type=int, default=256)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--d-model", type=int, default=256)
  parser.add_argument("--layers", type=int, default=4)
  parser.add_argument("--heads", type=int, default=8)
  parser.add_argument("--dropout", type=float, default=0.1)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--log-interval", type=int, default=25)
  parser.add_argument("--fused-optim", action="store_true")
  parser.add_argument("--no-jit-train-step", dest="jit_train_step", action="store_false")
  parser.add_argument("--no-jit-eval-step", dest="jit_eval_step", action="store_false")
  parser.add_argument("--max-train-batches", type=int, default=None)
  parser.add_argument("--max-eval-batches", type=int, default=None)
  parser.add_argument("--save", type=Path, default=Path("runs/atari-act-tinygrad-webgpu/model.safetensors"))
  parser.set_defaults(jit_train_step=True, jit_eval_step=True)
  return parser.parse_args()


if __name__ == "__main__":
  train(parse_args())
