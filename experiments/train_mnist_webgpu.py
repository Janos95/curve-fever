import argparse
import os
import time
from pathlib import Path

DEFAULT_WEBGPU_PATH = Path(
  "/Users/janosmeny/Projects/dawn-7069/out/Release/src/dawn/native/libwebgpu_dawn_tinygrad.dylib"
)

if DEFAULT_WEBGPU_PATH.exists():
  os.environ.setdefault("WEBGPU_PATH", str(DEFAULT_WEBGPU_PATH))
os.environ.setdefault("WEBGPU_BACKEND", "WGPUBackendType_Metal")

from tinygrad import Tensor, nn, dtypes  # noqa: E402
from tinygrad.nn.datasets import mnist  # noqa: E402
from tinygrad.nn.optim import Adam, AdamW  # noqa: E402
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save  # noqa: E402


class FourLayerCNN:
  def __init__(self):
    self.c1 = nn.Conv2d(1, 16, 3, padding=1)
    self.c2 = nn.Conv2d(16, 32, 3, padding=1)
    self.c3 = nn.Conv2d(32, 64, 3, padding=1)
    self.c4 = nn.Conv2d(64, 64, 3, padding=1)
    self.fc = nn.Linear(64 * 3 * 3, 10)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.c1(x).relu()
    x = self.c2(x).relu().max_pool2d(2)
    x = self.c3(x).relu().max_pool2d(2)
    x = self.c4(x).relu().max_pool2d(2)
    return self.fc(x.flatten(1))


def move_model(model: FourLayerCNN, device: str) -> None:
  Tensor.realize(*[tensor.to_(device) for tensor in get_state_dict(model).values()])


def load_mnist(device: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  x_train, y_train, x_test, y_test = mnist()
  x_train = (((x_train.cast(dtypes.float32) / 255.0) - 0.1307) / 0.3081).to(device).realize()
  x_test = (((x_test.cast(dtypes.float32) / 255.0) - 0.1307) / 0.3081).to(device).realize()
  y_train = y_train.cast(dtypes.int32).to(device).realize()
  y_test = y_test.cast(dtypes.int32).to(device).realize()
  return x_train, y_train, x_test, y_test


def evaluate(model: FourLayerCNN, x_test: Tensor, y_test: Tensor, batch_size: int) -> float:
  Tensor.training = False
  correct = 0
  total = x_test.shape[0]
  for start in range(0, total, batch_size):
    end = min(start + batch_size, total)
    pred = model(x_test[start:end]).argmax(axis=1)
    correct += int((pred == y_test[start:end]).sum().numpy())
  return correct / total


def train(args: argparse.Namespace) -> None:
  Tensor.manual_seed(args.seed)
  print(f"device={args.device} webgpu_path={os.environ.get('WEBGPU_PATH', '<unset>')}")
  x_train, y_train, x_test, y_test = load_mnist(args.device)
  model = FourLayerCNN()
  move_model(model, args.device)
  opt_cls = AdamW if args.weight_decay else Adam
  opt_kwargs = {"weight_decay": args.weight_decay} if args.weight_decay else {}
  opt = opt_cls(get_parameters(model), lr=args.lr, fused=args.fused_optim, **opt_kwargs)

  steps_per_epoch = x_train.shape[0] // args.batch_size
  best_acc = 0.0
  start_time = time.perf_counter()

  for epoch in range(1, args.max_epochs + 1):
    Tensor.training = True
    perm = Tensor.randperm(x_train.shape[0], device=args.device).realize()
    last_loss = None
    epoch_start = time.perf_counter()

    for step in range(steps_per_epoch):
      batch_idx = perm[step * args.batch_size:(step + 1) * args.batch_size]
      logits = model(x_train[batch_idx])
      loss = logits.sparse_categorical_crossentropy(y_train[batch_idx])
      opt.zero_grad()
      loss.backward()
      opt.step()
      if (step + 1) % args.log_interval == 0 or step + 1 == steps_per_epoch:
        last_loss = float(loss.numpy())
        print(f"epoch={epoch:02d} step={step + 1:03d}/{steps_per_epoch} loss={last_loss:.4f}", flush=True)

    acc = evaluate(model, x_test, y_test, args.eval_batch_size)
    best_acc = max(best_acc, acc)
    elapsed = time.perf_counter() - epoch_start
    total_elapsed = time.perf_counter() - start_time
    print(
      f"epoch={epoch:02d} loss={last_loss:.4f} test_acc={acc * 100:.2f}% "
      f"best={best_acc * 100:.2f}% epoch_s={elapsed:.1f} total_s={total_elapsed:.1f}",
      flush=True,
    )

    if acc >= args.target_acc:
      break

  if args.save:
    args.save.parent.mkdir(parents=True, exist_ok=True)
    Tensor.training = False
    state = {name: tensor.detach().to("CPU").realize() for name, tensor in get_state_dict(model).items()}
    safe_save(state, str(args.save))
    print(f"saved={args.save}")

  if best_acc < args.target_acc:
    raise SystemExit(f"target not reached: best={best_acc * 100:.2f}% target={args.target_acc * 100:.2f}%")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--device", default="WEBGPU")
  parser.add_argument("--target-acc", type=float, default=0.99)
  parser.add_argument("--max-epochs", type=int, default=10)
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--eval-batch-size", type=int, default=512)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--fused-optim", action="store_true")
  parser.add_argument("--seed", type=int, default=1337)
  parser.add_argument("--log-interval", type=int, default=50)
  parser.add_argument("--save", type=Path, default=Path("runs/mnist_webgpu_4layer_99.safetensors"))
  return parser.parse_args()


if __name__ == "__main__":
  train(parse_args())
