"""PyTorch port of the handwriting-synthesis network (Graves 2013, as implemented by sjvasquez).

Three stacked LSTMs with a soft attention window over the one-hot text, and a mixture density
output (20 bivariate Gaussians + end-of-stroke Bernoulli). Mirrors the TF1 graph exactly:

- ``TFLSTMCell`` keeps TensorFlow's weight layout (kernel ``[in+H, 4H]``, gates i, j, f, o,
  forget bias +1), so converted checkpoint arrays load with no reshuffling.
- Attention: ``kappa += softplus(.)/25``, ``phi(u) = sum alpha * exp(-(kappa-u)^2 / beta)``
  (the reference divides by beta where the paper multiplies), beta clipped at 0.01.
- Bias (neatness): mixture logits ``* (1 + b)``, log-sigmas ``- b``; components and
  end-of-stroke probabilities under 0.01 are zeroed, as in the reference sampler.

Units: inputs/outputs are normalized offsets ``(dx, dy, eos)`` with +y **up** (IAM style).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ALPHABET_SIZE = 73


class TFLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.kernel = nn.Parameter(torch.empty(input_size + hidden, 4 * hidden))
        self.bias = nn.Parameter(torch.zeros(4 * hidden))
        nn.init.xavier_uniform_(self.kernel)

    def forward(self, x, h, c):
        z = torch.cat([x, h], dim=1) @ self.kernel + self.bias
        i, j, f, o = z.chunk(4, dim=1)
        c = torch.sigmoid(f + 1.0) * c + torch.sigmoid(i) * torch.tanh(j)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class Dense(nn.Module):
    """TF-style dense layer: ``x @ weights + biases`` with weights ``[in, out]``."""

    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_in, n_out))
        self.biases = nn.Parameter(torch.zeros(n_out))
        nn.init.kaiming_uniform_(self.weights.T, a=5 ** 0.5)

    def forward(self, x):
        return x @ self.weights + self.biases


@dataclass
class State:
    h1: torch.Tensor
    c1: torch.Tensor
    h2: torch.Tensor
    c2: torch.Tensor
    h3: torch.Tensor
    c3: torch.Tensor
    kappa: torch.Tensor
    w: torch.Tensor
    phi: torch.Tensor

    def where(self, keep: torch.Tensor, other: "State") -> "State":
        """Per batch row: take ``self`` where ``keep`` (B,) is True, else ``other``."""
        k = keep.view(-1, 1)
        return State(*[torch.where(k, a, b) for a, b in zip(self.__dict__.values(), other.__dict__.values())])


@dataclass
class Mixture:
    pi: torch.Tensor     # (B, K)
    mu1: torch.Tensor
    mu2: torch.Tensor
    s1: torch.Tensor
    s2: torch.Tensor
    rho: torch.Tensor
    e: torch.Tensor      # (B, 1)


class HandwritingRNN(nn.Module):
    def __init__(self, lstm_size: int = 400, attn_components: int = 10, out_components: int = 20,
                 alphabet_size: int = ALPHABET_SIZE) -> None:
        super().__init__()
        H, A = lstm_size, alphabet_size
        self.H, self.A, self.K_attn, self.K_out = H, A, attn_components, out_components
        self.lstm1 = TFLSTMCell(A + 3, H)
        self.attention = Dense(A + 3 + H, 3 * attn_components)
        self.lstm2 = TFLSTMCell(3 + H + A, H)
        self.lstm3 = TFLSTMCell(3 + H + A, H)
        self.gmm = Dense(H, 6 * out_components + 1)

    # ------------------------------------------------------------------ core

    def zero_state(self, batch: int, char_len: int, device=None) -> State:
        z = lambda n: torch.zeros(batch, n, device=device)  # noqa: E731
        return State(z(self.H), z(self.H), z(self.H), z(self.H), z(self.H), z(self.H),
                     z(self.K_attn), z(self.A), z(char_len))

    def step(self, x: torch.Tensor, s: State, chars: torch.Tensor, char_mask: torch.Tensor) -> State:
        """One timestep. ``chars``: (B, U, A) one-hot; ``char_mask``: (B, U) float."""
        h1, c1 = self.lstm1(torch.cat([s.w, x], 1), s.h1, s.c1)
        params = F.softplus(self.attention(torch.cat([s.w, x, h1], 1)))
        alpha, beta, dkappa = params.chunk(3, dim=1)
        kappa = s.kappa + dkappa / 25.0
        beta = beta.clamp(min=0.01)
        u = torch.arange(chars.shape[1], device=x.device, dtype=x.dtype).view(1, 1, -1)
        phi = (alpha.unsqueeze(2) * torch.exp(-((kappa.unsqueeze(2) - u) ** 2) / beta.unsqueeze(2))).sum(1)
        w = (phi.unsqueeze(2) * chars * char_mask.unsqueeze(2)).sum(1)
        h2, c2 = self.lstm2(torch.cat([x, h1, w], 1), s.h2, s.c2)
        h3, c3 = self.lstm3(torch.cat([x, h2, w], 1), s.h3, s.c3)
        return State(h1, c1, h2, c2, h3, c3, kappa, w, phi)

    def mixture(self, h3: torch.Tensor, bias: torch.Tensor | None = None, sampling: bool = False) -> Mixture:
        K = self.K_out
        z = self.gmm(h3)
        pi, sig, rho, mu, e = torch.split(z, [K, 2 * K, K, 2 * K, 1], dim=-1)
        if bias is not None:
            b = bias.view(-1, 1)
            pi = pi * (1 + b)
            sig = sig - b
        pi = torch.softmax(pi, dim=-1)
        sig = torch.exp(sig).clamp(min=1e-4)
        rho = torch.tanh(rho).clamp(-1 + 1e-8, 1 - 1e-8)
        e = torch.sigmoid(e).clamp(1e-8, 1 - 1e-8)
        if sampling:
            pi = torch.where(pi < 0.01, torch.zeros_like(pi), pi)
            pi = pi / pi.sum(-1, keepdim=True)
            e = torch.where(e < 0.01, torch.zeros_like(e), e)
        mu1, mu2 = mu.chunk(2, dim=-1)
        s1, s2 = sig.chunk(2, dim=-1)
        return Mixture(pi, mu1, mu2, s1, s2, rho, e)

    # ------------------------------------------------------------------ training

    def nll(self, x: torch.Tensor, y: torch.Tensor, lengths: torch.Tensor,
            chars: torch.Tensor, char_mask: torch.Tensor) -> torch.Tensor:
        """Mean negative log-likelihood per valid timestep (teacher forcing).

        ``x``/``y``: (B, T, 3) inputs and next-step targets; ``lengths``: (B,).
        """
        B, T, _ = x.shape
        s = self.zero_state(B, chars.shape[1], x.device)
        hs = []
        for t in range(T):
            s = self.step(x[:, t], s, chars, char_mask)
            hs.append(s.h3)
        m = self.mixture(torch.stack(hs, 1))
        y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3]
        n1, n2 = (y1 - m.mu1) / m.s1, (y2 - m.mu2) / m.s2
        one_m_r2 = 1 - m.rho ** 2
        zz = n1 ** 2 + n2 ** 2 - 2 * m.rho * n1 * n2
        log_n = -zz / (2 * one_m_r2) - torch.log(2 * np.pi * m.s1 * m.s2 * torch.sqrt(one_m_r2))
        log_gmm = torch.logsumexp(torch.log(m.pi.clamp(min=1e-12)) + log_n, dim=-1)
        log_b = torch.where(y3 == 1, torch.log(m.e), torch.log(1 - m.e)).squeeze(-1)
        nll = -(log_gmm + log_b)
        mask = torch.arange(T, device=x.device).view(1, -1) < lengths.view(-1, 1)
        return (nll * mask).sum() / mask.sum().clamp(min=1)

    # ------------------------------------------------------------------ sampling

    @torch.no_grad()
    def sample(
        self,
        chars: torch.Tensor,          # (B, U) int, padded with 0
        char_lens: torch.Tensor,      # (B,)
        bias: torch.Tensor,           # (B,)
        prime: torch.Tensor | None = None,       # (B, P, 3)
        prime_lens: torch.Tensor | None = None,  # (B,)
        max_steps: int = 2000,
        generator: torch.Generator | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate strokes. Returns per row ``(offsets (N, 3), char_index_per_step (N,))``."""
        B, U = chars.shape
        onehot = F.one_hot(chars, self.A).float()
        mask = (torch.arange(U).view(1, -1) < char_lens.view(-1, 1)).float()
        s = self.zero_state(B, U)

        if prime is not None:
            for t in range(prime.shape[1]):
                nxt = self.step(prime[:, t], s, onehot, mask)
                s = nxt.where(t < prime_lens, s)
            x = self._draw(self.mixture(s.h3, bias, sampling=True), generator)
        else:
            x = torch.tensor([[0.0, 0.0, 1.0]]).repeat(B, 1)

        outs, idxs = [], []
        done = torch.zeros(B, dtype=torch.bool)
        for _ in range(max_steps):
            s = self.step(x, s, onehot, mask)
            x = self._draw(self.mixture(s.h3, bias, sampling=True), generator)
            char_idx = s.phi.argmax(1)
            outs.append(torch.where(done.view(-1, 1), torch.zeros_like(x), x))
            idxs.append(char_idx)
            final = char_idx >= char_lens - 1
            past = char_idx >= char_lens
            done = done | (final & (x[:, 2] == 1)) | past
            if bool(done.all()):
                break

        out = torch.stack(outs, 1).numpy()
        idx = torch.stack(idxs, 1).numpy()
        result = []
        for b in range(B):
            keep = ~np.all(out[b] == 0.0, axis=1)
            result.append((out[b][keep], idx[b][keep]))
        return result

    @staticmethod
    def _draw(m: Mixture, gen: torch.Generator | None) -> torch.Tensor:
        B = m.pi.shape[0]
        k = torch.multinomial(m.pi, 1, generator=gen)
        pick = lambda t: t.gather(1, k)  # noqa: E731
        mu1, mu2, s1, s2, rho = map(pick, (m.mu1, m.mu2, m.s1, m.s2, m.rho))
        z1 = torch.randn(B, 1, generator=gen)
        z2 = torch.randn(B, 1, generator=gen)
        x1 = mu1 + s1 * z1
        x2 = mu2 + s2 * (rho * z1 + torch.sqrt(1 - rho ** 2) * z2)
        e = torch.bernoulli(m.e, generator=gen)
        return torch.cat([x1, x2, e], 1)

    # ------------------------------------------------------------------ io

    @classmethod
    def from_npz(cls, path) -> "HandwritingRNN":
        data = np.load(path, allow_pickle=False)
        H = data["lstm1.bias"].shape[0] // 4
        net = cls(lstm_size=H, attn_components=data["attention.biases"].shape[0] // 3,
                  out_components=(data["gmm.biases"].shape[0] - 1) // 6)
        state = {k: torch.from_numpy(data[k].astype(np.float32)) for k in data.files}
        net.load_state_dict(state)
        return net.eval()
