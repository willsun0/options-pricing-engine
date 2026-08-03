r"""A neural network fit to the implied volatility surface, and what it costs.

===============================================================================
WHAT THIS MODEL IS, AND WHAT IT IS NOT
===============================================================================

This is a small feed-forward network trained to reproduce the implied volatility
surface of Phase 4:

    (strike, time to expiry, moneyness)  ->  implied volatility

It is worth being blunt about what that does and does not achieve, because
"neural network for option pricing" invites two opposite misreadings.

**It is not a pricing model.** It contains no dynamics, no stochastic process, no
risk-neutral measure. It never predicts what volatility *will be*. It is a
smooth, flexible **interpolator** fitted to quotes that already exist. Every
number it produces is still fed into Black-Scholes to get a price — the network
supplies the volatility argument, nothing more. Anything it "knows" about
markets, it read off today's quotes.

**Nor is it pointless.** It is a genuinely better interpolator than SVI in one
specific and useful way: SVI fits each expiry in isolation, so a five-parameter
curve is calibrated to the 11-day slice with no knowledge of the 28-day slice.
The network sees the whole surface at once and shares parameters across
expiries, so a sparse slice borrows shape from its neighbours. On this data the
179-day expiry has only 36 usable quotes — SVI fits five parameters to 36 noisy
points, while the network fits that region with support from the 1000-odd quotes
around it. Fitting the surface jointly is the real contribution here, not
"nonlinearity".

-------------------------------------------------------------------------------
WHY BLACK-SCHOLES CANNOT DO THIS
-------------------------------------------------------------------------------

Black-Scholes assumes one constant sigma. It therefore predicts implied
volatility is a **horizontal plane**: the same number at every strike and every
expiry. Phase 4 measured the actual surface, and it is not flat — it slopes down
in strike (skew) and up in expiry (term structure).

The gap is not a calibration failure. It is the model's assumptions being false:
real returns have fat tails and negative skewness, volatility is itself
stochastic and correlated with spot (the leverage effect), and the market charges
a premium for crash protection. A constant-vol model has no parameter that can
express any of that, so the entire discrepancy is pushed into the one input that
is free to vary — and *that displacement is what the smile is*.

The network does not fix Black-Scholes. It **describes the displacement**. It is
a map of where and by how much the model is wrong, which is exactly what the
implied vol surface has always been; the network just supplies a smooth,
differentiable, jointly-fitted version of that map.

-------------------------------------------------------------------------------
ARCHITECTURE, AND WHY IT IS DELIBERATELY SMALL
-------------------------------------------------------------------------------

Three inputs, two hidden layers of 64 units, one output: about 4,500 parameters
against roughly 1,000 training quotes. That is four parameters per data point,
which sounds reckless and is the reason early stopping below is load-bearing
rather than decorative. Reaching for a deeper network would still be a mistake —
the target is a smooth two-dimensional surface, not ImageNet.

**The activation is tanh, not ReLU, and this is the single most important choice
in the file.** ReLU networks are piecewise linear, so the surface they produce
has kinks and a second derivative that is zero almost everywhere and undefined at
the joints. That is fatal for this application, because by Breeden-Litzenberger
the risk-neutral density is the *second derivative* of call price in strike:

    q(K) = e^{rT} d^2C/dK^2

A kinked volatility surface produces a density made of spikes and gaps.

This is measured, not assumed, and the measurement makes a sharper point than the
argument does. Fitting the same SPY surface with each activation, at the defaults
below, and measuring curvature along the busiest expiry slice:

    activation   validation RMSE     peak |d2 sigma/dk2|
    tanh              0.168 vp              15.0
    ReLU              0.151 vp             230.6
    SiLU              0.238 vp              12.5

**ReLU fits marginally better on the metric everyone reports, and is fifteen
times worse on the one that determines whether the implied density is usable.**
Judging these three by RMSE alone would actively select the broken one. SiLU —
smooth, but otherwise ReLU-shaped — is as well behaved as tanh despite the worst
RMSE of the three, which confirms the mechanism is the activation's *smoothness*
and not anything else about its shape. :class:`Activation` keeps all three so the
Phase 5 figure can show the kinks rather than assert them; run
``phase5_ml_vol_surface.py`` to reproduce this table.

**Output goes through softplus** so predicted volatility is positive by
construction. An unconstrained linear output can and does emit negative
volatilities when extrapolating, which are not merely wrong but meaningless.

**Training is full-batch.** With ~1,000 points there is no reason to minibatch,
and removing the shuffling makes runs bit-for-bit reproducible given a seed —
worth more here than any speed gain.

-------------------------------------------------------------------------------
THE THREE THINGS KEEPING IT SMOOTH
-------------------------------------------------------------------------------

A network with four parameters per data point could in principle interpolate
bid-ask noise, producing a surface that wiggles between adjacent strikes. Three
mechanisms are available to prevent that — and measuring them individually
overturned the ordering this section originally asserted.

The textbook answer is that **weight decay** is the smoothness control: large
first-layer weights let a tanh unit act as a sharp switch, so bounding them
bounds curvature. Sweeping it on this surface says otherwise:

    weight decay    validation RMSE     mean |d2 sigma/dk2|
    0                    0.115 vp              0.66
    1e-6                 0.168 vp              0.58
    1e-5                 0.519 vp              0.53
    1e-4                 1.105 vp              0.49
    1e-3                 2.207 vp              0.53

Raising it by four orders of magnitude degrades the fit **twentyfold** and
reduces roughness by about 20%. It is not buying smoothness; it is buying
underfitting, which merely resembles smoothness because a flat surface is smooth
too. The default is therefore a token 1e-6 — enough to keep weights from
wandering during the long tail of training, cheap enough to cost little fit.

What actually keeps the surface smooth is the other two:

1. **The smooth activation**, which bounds curvature structurally rather than
   statistically — see the table above, where it moves peak curvature by 39x.
2. **Early stopping** on a held-out split. The network fits the broad shape of
   the surface long before it fits the noise, so stopping at the best validation
   loss captures the signal and discards the memorisation. With this
   parameter-to-data ratio it is the only thing standing between the model and
   memorising the bid-ask spread.

Note what is *not* here: any no-arbitrage constraint. SVI's parameters can be
confined to a region proven arbitrage-free. The network has no such guarantee,
and nothing stops it from implying a negative density. Phase 5 checks this after
the fact with the same Durrleman function used for SVI. That asymmetry —
guarantee versus inspection — is the honest cost of the flexibility.

-------------------------------------------------------------------------------
INTERPOLATION VERSUS EXTRAPOLATION
-------------------------------------------------------------------------------

The critical limitation, and the one the Phase 5 figures are built to show.

**Inside the convex hull of the training data** the network is excellent. It has
quotes on all sides and is doing what it is good at: smoothing.

**Outside that hull it has no defensible behaviour at all.** A tanh network
saturates: every unit flattens to +/-1, so far from the data the output tends to
a constant that depends on nothing but the weights. The surface simply goes flat
at whatever level the saturation happens to produce. It is not that the
extrapolation is inaccurate — it is that the functional form encodes no opinion
about the region, and the number it returns is an artefact of the architecture.

The usual next sentence is that SVI is safe here because Lee's moment formula
forces its wings to be linear in total variance, so it is wrong in a constrained,
knowable way rather than an arbitrary one. Training both on the middle 60% of
log-moneyness and scoring them on the wings neither model saw:

    model    RMSE inside band      RMSE in the wings      degradation
    network      0.182 vp               3.243 vp              17.8x
    SVI          0.105 vp               2.120 vp              20.3x

The theory earns SVI a real but modest edge — about 1.5x lower error in the
wings. What it does **not** earn is safety: SVI degrades by the same factor, and
2.1 vol points of error is not a number anyone should quote a price from. The
honest conclusion is that *neither* model extrapolates, and the interesting
difference is not accuracy but diagnosability — SVI's failure shows up as
Durrleman's ``g(k)`` going negative, which is checkable in closed form, whereas
the network's has no comparable tell. (The two interior figures are not strictly
comparable: the network's is on held-out quotes, SVI's is in-sample across six
independently fitted slices. The wings column, which is out-of-sample for both,
is the like-for-like comparison.)

:func:`extrapolation_report` reproduces the network row; the Phase 5 notebook
adds the SVI row.

References: Hutchinson, Lo & Poggio, *A nonparametric approach to pricing and
hedging derivative securities via learning networks* (1994) — the original, and
still the clearest statement of the idea; Ferguson & Green (2018); Horvath,
Muguruza & Tomas, *Deep learning volatility* (2021); Breeden & Litzenberger
(1978).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

__all__ = [
    "Activation",
    "FeatureScaler",
    "TrainingConfig",
    "TrainingHistory",
    "VolatilityNet",
    "VolatilitySurfaceModel",
    "build_features",
    "train_surface_model",
    "extrapolation_report",
    "FEATURE_NAMES",
]

# The three inputs, in the order every array in this module uses. Fixed as a
# module constant rather than passed around, because silently transposing two
# feature columns between training and prediction produces a model that trains
# fine and predicts nonsense.
FEATURE_NAMES: tuple[str, str, str] = ("strike", "time_to_expiry", "moneyness")


class Activation(str, Enum):
    """Hidden-layer nonlinearity.

    ``TANH`` is the default and the right choice for a volatility surface: it is
    smooth, so the fitted surface has continuous derivatives of every order and a
    well-defined implied density. ``RELU`` is provided for comparison — it trains
    slightly faster and fits about as well by RMSE, while producing a visibly
    kinked surface. ``SILU`` is a smooth ReLU-like alternative, included because
    it is the modern default elsewhere and it is useful to see that the smoothness
    argument, not the shape, is what matters.
    """

    TANH = "tanh"
    RELU = "relu"
    SILU = "silu"

    def module(self) -> nn.Module:
        """Return a fresh ``torch`` module for this activation.

        Returns:
            The corresponding activation layer.
        """
        return {
            Activation.TANH: nn.Tanh,
            Activation.RELU: nn.ReLU,
            Activation.SILU: nn.SiLU,
        }[self]()

    @property
    def is_smooth(self) -> bool:
        """Whether the activation has continuous derivatives of all orders.

        Returns:
            ``True`` for tanh and SiLU, ``False`` for ReLU.
        """
        return self is not Activation.RELU


def build_features(
    strikes: NDArray[np.float64],
    times_to_expiry: NDArray[np.float64],
    moneyness: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Assemble the ``(n, 3)`` feature matrix in the canonical column order.

    The three inputs are not independent — moneyness is ``strike / forward`` and
    the forward depends on expiry — so strike is partially redundant given the
    other two. It is kept because the redundancy is only partial (the forward
    varies with expiry, so the map is not one-to-one) and because a model that
    takes strike directly is easier to call from pricing code that already has a
    strike in hand.

    Args:
        strikes: Strike prices, shape ``(n,)``.
        times_to_expiry: Times to expiry in years, shape ``(n,)``.
        moneyness: Strike over forward, ``K/F``, shape ``(n,)``.

    Returns:
        Feature matrix of shape ``(n, 3)`` with columns :data:`FEATURE_NAMES`.

    Raises:
        ValueError: If the inputs do not all have the same 1-D shape.
    """
    columns = [np.asarray(x, dtype=float) for x in (strikes, times_to_expiry, moneyness)]
    if any(column.ndim != 1 for column in columns):
        raise ValueError("all feature inputs must be 1-D")
    if len({column.shape for column in columns}) != 1:
        raise ValueError(
            "strikes, times_to_expiry and moneyness must have the same length, got "
            + ", ".join(str(column.shape) for column in columns)
        )
    return np.column_stack(columns)


@dataclass(frozen=True)
class FeatureScaler:
    """Per-column standardisation, fitted on the training set only.

    Standardising is not cosmetic for a tanh network. Strike is order 700, time to
    expiry is order 0.1, and moneyness is order 1. Fed raw, the first layer's
    tanh units saturate immediately on the strike column and the gradient with
    respect to the other two is swamped — the network trains to a constant.

    The statistics come from the training split alone. Computing them over the
    full dataset would leak the validation set's location and scale into the
    model, which quietly flatters every validation number reported afterwards.

    Attributes:
        mean: Per-column means, shape ``(3,)``.
        scale: Per-column standard deviations, shape ``(3,)``, floored away from
            zero so a constant column cannot divide by zero.
    """

    mean: NDArray[np.float64]
    scale: NDArray[np.float64]

    @classmethod
    def fit(cls, features: NDArray[np.float64]) -> FeatureScaler:
        """Compute standardisation statistics from a feature matrix.

        Args:
            features: Training features, shape ``(n, d)``.

        Returns:
            A fitted scaler.
        """
        array = np.asarray(features, dtype=float)
        scale = array.std(axis=0)
        # A column with no variation carries no information; scaling it by 1
        # leaves it at its (now zero) centred value rather than producing NaN.
        scale = np.where(scale < 1e-12, 1.0, scale)
        return cls(mean=array.mean(axis=0), scale=scale)

    def transform(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Standardise a feature matrix using the fitted statistics.

        Args:
            features: Features to transform, shape ``(n, d)``.

        Returns:
            Standardised features of the same shape.
        """
        return (np.asarray(features, dtype=float) - self.mean) / self.scale


class VolatilityNet(nn.Module):
    """A small MLP mapping standardised features to a positive volatility.

    Attributes:
        network: The underlying ``torch`` sequential stack.
    """

    def __init__(
        self,
        n_features: int = 3,
        hidden_sizes: tuple[int, ...] = (32, 32),
        activation: Activation = Activation.TANH,
    ) -> None:
        """Build the network.

        Args:
            n_features: Number of input features.
            hidden_sizes: Width of each hidden layer.
            activation: Hidden-layer nonlinearity. See :class:`Activation` for why
                the default is tanh rather than ReLU.

        Raises:
            ValueError: If ``hidden_sizes`` is empty or contains a non-positive
                width.
        """
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer")
        if any(size <= 0 for size in hidden_sizes):
            raise ValueError(f"hidden layer widths must be positive, got {hidden_sizes}")

        layers: list[nn.Module] = []
        in_features = n_features
        for width in hidden_sizes:
            layers.append(nn.Linear(in_features, width))
            layers.append(activation.module())
            in_features = width
        layers.append(nn.Linear(in_features, 1))
        # Softplus rather than a bare linear output: implied volatility is
        # positive by definition, and an unconstrained head returns negative
        # values when extrapolating past the training data.
        layers.append(nn.Softplus())
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict volatility for a batch of standardised features.

        Args:
            features: Standardised features, shape ``(n, n_features)``.

        Returns:
            Predicted volatilities, shape ``(n,)``.
        """
        return self.network(features).squeeze(-1)


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for :func:`train_surface_model`.

    Attributes:
        hidden_sizes: Hidden layer widths.
        activation: Hidden-layer nonlinearity.
        epochs: Maximum full-batch gradient steps. The default is large because
            full-batch Adam on 1,000 points takes a few hundred microseconds per
            step, and the fit was measured still improving at epoch 11,000 —
            stopping at the more usual few thousand left real accuracy on the
            table. ``patience`` ends the run long before this cap in practice.
        learning_rate: Adam step size. 1e-2 is large by deep-learning standards
            and appropriate here: the problem is tiny, full-batch, and smooth.
        weight_decay: L2 penalty on the weights. Kept token-sized deliberately —
            see the module docstring for the sweep showing that larger values
            trade fit away without buying smoothness.
        validation_fraction: Share of quotes held out for early stopping.
        patience: Stop after this many epochs without a new best validation loss.
            ``None`` disables early stopping and trains the full ``epochs``.
        seed: Seeds weight initialisation and the train/validation split, so a
            run is reproducible end to end.
    """

    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: Activation = Activation.TANH
    epochs: int = 40_000
    learning_rate: float = 1e-2
    weight_decay: float = 1e-6
    validation_fraction: float = 0.2
    patience: int | None = 2_000
    seed: int = 0


@dataclass
class TrainingHistory:
    """Per-epoch losses and the epoch the returned weights came from.

    Losses are mean squared error in volatility units, so their square root is
    directly comparable to the RMSE that :class:`~options_engine.vol_surface.svi.SVIFitResult`
    reports.

    Attributes:
        train_loss: Training MSE at each epoch.
        validation_loss: Validation MSE at each epoch. Empty if no validation
            split was requested.
        best_epoch: Epoch whose weights were restored, or the final epoch if
            early stopping was disabled.
        stopped_early: Whether training halted before ``epochs`` was reached.
    """

    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False

    def best_validation_rmse(self) -> float:
        """Return the best validation RMSE in volatility units.

        Returns:
            ``sqrt`` of the lowest validation MSE, or ``nan`` if there was no
            validation split.
        """
        if not self.validation_loss:
            return float("nan")
        return float(np.sqrt(min(self.validation_loss)))


@dataclass(frozen=True)
class VolatilitySurfaceModel:
    """A trained network bundled with the scaler its inputs must pass through.

    Keeping the two together is the point of the class. A network trained on
    standardised features and then handed raw ones does not fail loudly — it
    returns plausible-looking volatilities that are simply wrong.

    Attributes:
        net: The trained network.
        scaler: The scaler fitted on the training features.
        config: The configuration used to train it.
        training_hull: Per-feature ``(min, max)`` of the training data, used by
            :meth:`is_in_training_range` to flag extrapolation.
    """

    net: VolatilityNet
    scaler: FeatureScaler
    config: TrainingConfig
    training_hull: NDArray[np.float64]

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict implied volatility for a feature matrix.

        Args:
            features: Raw (unstandardised) features, shape ``(n, 3)`` with columns
                :data:`FEATURE_NAMES`.

        Returns:
            Predicted implied volatilities, shape ``(n,)``.
        """
        standardised = self.scaler.transform(np.atleast_2d(features))
        self.net.eval()
        with torch.no_grad():
            predictions = self.net(torch.as_tensor(standardised, dtype=torch.float32))
        return predictions.numpy().astype(float)

    def is_in_training_range(self, features: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Flag rows that fall inside the per-feature range seen during training.

        This is a bounding box, not a convex hull, so it is permissive: a point
        can pass this check and still sit in a gap between training quotes. It is
        the cheap version of the right question, and a point that fails it is
        unambiguously an extrapolation.

        Args:
            features: Raw features, shape ``(n, 3)``.

        Returns:
            Boolean mask of shape ``(n,)``, ``True`` where every feature is within
            its training range.
        """
        array = np.atleast_2d(np.asarray(features, dtype=float))
        inside = (array >= self.training_hull[0]) & (array <= self.training_hull[1])
        return np.all(inside, axis=1)

    def save(self, path: Path) -> None:
        """Persist the model, scaler and config to a single file.

        The payload is deliberately restricted to tensors and primitives — no
        pickled dataclass, no numpy array. That is what lets :meth:`load` pass
        ``weights_only=True``, which refuses to execute arbitrary code while
        unpickling. Storing the ``TrainingConfig`` object directly would be
        shorter and would force the load path to disable that protection.

        Args:
            path: Destination file. Parent directories are created if needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "mean": torch.as_tensor(self.scaler.mean, dtype=torch.float64),
                "scale": torch.as_tensor(self.scaler.scale, dtype=torch.float64),
                "training_hull": torch.as_tensor(self.training_hull, dtype=torch.float64),
                "hidden_sizes": list(self.config.hidden_sizes),
                "activation": self.config.activation.value,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "validation_fraction": self.config.validation_fraction,
                "patience": self.config.patience,
                "seed": self.config.seed,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> VolatilitySurfaceModel:
        """Load a model previously written by :meth:`save`.

        Args:
            path: File to read.

        Returns:
            The restored model, ready for :meth:`predict`.
        """
        payload = torch.load(Path(path), weights_only=True)
        config = TrainingConfig(
            hidden_sizes=tuple(payload["hidden_sizes"]),
            activation=Activation(payload["activation"]),
            epochs=payload["epochs"],
            learning_rate=payload["learning_rate"],
            weight_decay=payload["weight_decay"],
            validation_fraction=payload["validation_fraction"],
            patience=payload["patience"],
            seed=payload["seed"],
        )
        mean = payload["mean"].numpy()
        net = VolatilityNet(
            n_features=len(mean),
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
        )
        net.load_state_dict(payload["state_dict"])
        net.eval()
        return cls(
            net=net,
            scaler=FeatureScaler(mean=mean, scale=payload["scale"].numpy()),
            config=config,
            training_hull=payload["training_hull"].numpy(),
        )


def _split_indices(
    n_samples: int, validation_fraction: float, seed: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Split row indices into training and validation sets.

    The split is random rather than stratified by expiry. That is the right
    default for measuring *interpolation* quality, which is what the headline
    number is about: every held-out quote has training quotes on either side of
    it. Measuring extrapolation needs a deliberately different split, which is
    what :func:`extrapolation_report` constructs.

    Args:
        n_samples: Number of rows.
        validation_fraction: Share to hold out, in ``[0, 1)``.
        seed: Seed for the permutation.

    Returns:
        A ``(train_indices, validation_indices)`` tuple.
    """
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_samples)
    n_validation = int(round(validation_fraction * n_samples))
    return permutation[n_validation:], permutation[:n_validation]


def train_surface_model(
    features: NDArray[np.float64],
    implied_vols: NDArray[np.float64],
    config: TrainingConfig | None = None,
) -> tuple[VolatilitySurfaceModel, TrainingHistory]:
    """Train the volatility surface network.

    Args:
        features: Raw features, shape ``(n, 3)`` with columns :data:`FEATURE_NAMES`.
        implied_vols: Target implied volatilities, shape ``(n,)``.
        config: Hyperparameters. Defaults to :class:`TrainingConfig`.

    Returns:
        A ``(model, history)`` tuple. The model's weights are those from the best
        validation epoch when early stopping is enabled, not the final epoch.

    Raises:
        ValueError: If the shapes are inconsistent, the data is non-finite, the
            targets are not positive, or the validation fraction leaves an empty
            training set.
    """
    config = config or TrainingConfig()

    feature_array = np.asarray(features, dtype=float)
    targets = np.asarray(implied_vols, dtype=float)

    if feature_array.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {feature_array.shape}")
    if targets.shape != (feature_array.shape[0],):
        raise ValueError(
            f"implied_vols {targets.shape} must be 1-D of length {feature_array.shape[0]}"
        )
    if not np.all(np.isfinite(feature_array)) or not np.all(np.isfinite(targets)):
        raise ValueError("features and implied_vols must be finite")
    if np.any(targets <= 0.0):
        raise ValueError("implied_vols must be strictly positive")
    if not 0.0 <= config.validation_fraction < 1.0:
        raise ValueError(
            f"validation_fraction must be in [0, 1), got {config.validation_fraction}"
        )

    train_index, validation_index = _split_indices(
        feature_array.shape[0], config.validation_fraction, config.seed
    )
    if train_index.size == 0:
        raise ValueError("validation_fraction leaves no training data")

    # Fit the scaler on the training rows only — see FeatureScaler for why.
    scaler = FeatureScaler.fit(feature_array[train_index])

    train_x = torch.as_tensor(scaler.transform(feature_array[train_index]), dtype=torch.float32)
    train_y = torch.as_tensor(targets[train_index], dtype=torch.float32)
    has_validation = validation_index.size > 0
    if has_validation:
        validation_x = torch.as_tensor(
            scaler.transform(feature_array[validation_index]), dtype=torch.float32
        )
        validation_y = torch.as_tensor(targets[validation_index], dtype=torch.float32)

    # Seed immediately before constructing the net so weight initialisation is
    # reproducible regardless of what else has touched the global RNG.
    torch.manual_seed(config.seed)
    net = VolatilityNet(
        n_features=feature_array.shape[1],
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
    )
    optimiser = torch.optim.Adam(
        net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.MSELoss()

    history = TrainingHistory()
    best_loss = float("inf")
    best_state = {key: value.clone() for key, value in net.state_dict().items()}
    epochs_since_best = 0

    for epoch in range(config.epochs):
        net.train()
        optimiser.zero_grad()
        loss = loss_fn(net(train_x), train_y)
        loss.backward()
        optimiser.step()
        history.train_loss.append(float(loss.detach()))

        if not has_validation:
            continue

        net.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(net(validation_x), validation_y))
        history.validation_loss.append(validation_loss)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.clone() for key, value in net.state_dict().items()}
            history.best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if config.patience is not None and epochs_since_best >= config.patience:
                history.stopped_early = True
                break

    if has_validation:
        # Restore the best-validation weights. Without this, early stopping only
        # decides *when* to stop and still returns weights that have already
        # started overfitting.
        net.load_state_dict(best_state)
    else:
        history.best_epoch = len(history.train_loss) - 1
    net.eval()

    hull = np.vstack([feature_array.min(axis=0), feature_array.max(axis=0)])
    return VolatilitySurfaceModel(net=net, scaler=scaler, config=config, training_hull=hull), history


@dataclass(frozen=True)
class ExtrapolationReport:
    """Errors inside versus outside the moneyness band the model was trained on.

    Attributes:
        interior_rmse: RMSE in volatility units on held-out quotes *inside* the
            training band — the interpolation number.
        exterior_rmse: RMSE on quotes in the wings, which the model never saw —
            the extrapolation number.
        n_interior: Number of held-out interior quotes.
        n_exterior: Number of exterior quotes.
        band: The ``(low, high)`` log-moneyness band used for training.
    """

    interior_rmse: float
    exterior_rmse: float
    n_interior: int
    n_exterior: int
    band: tuple[float, float]

    @property
    def degradation_factor(self) -> float:
        """How many times worse extrapolation is than interpolation.

        Returns:
            ``exterior_rmse / interior_rmse``.
        """
        return self.exterior_rmse / self.interior_rmse


def extrapolation_report(
    features: NDArray[np.float64],
    implied_vols: NDArray[np.float64],
    log_moneyness: NDArray[np.float64],
    quantile: float = 0.2,
    config: TrainingConfig | None = None,
) -> ExtrapolationReport:
    """Quantify how badly the model degrades outside its training range.

    Trains on the central band of log-moneyness only, then scores the wings it
    never saw. This is the honest test that a random train/validation split
    cannot give: under a random split every held-out point is surrounded by
    training points, so the reported error measures interpolation and says
    nothing about the wings — which is precisely where a volatility surface is
    hardest and where the model would be used to price an untraded strike.

    Args:
        features: Raw features, shape ``(n, 3)``.
        implied_vols: Target implied volatilities, shape ``(n,)``.
        log_moneyness: Log-moneyness of each quote, shape ``(n,)``, used only to
            define the band.
        quantile: Fraction trimmed from each end. 0.2 trains on the middle 60%.
        config: Hyperparameters for the retrained model.

    Returns:
        The comparison of interior and exterior errors.

    Raises:
        ValueError: If ``quantile`` is not in ``(0, 0.5)`` or the split leaves one
            side empty.
    """
    if not 0.0 < quantile < 0.5:
        raise ValueError(f"quantile must be in (0, 0.5), got {quantile}")

    config = config or TrainingConfig()
    feature_array = np.asarray(features, dtype=float)
    targets = np.asarray(implied_vols, dtype=float)
    k = np.asarray(log_moneyness, dtype=float)

    low, high = np.quantile(k, [quantile, 1.0 - quantile])
    interior = (k >= low) & (k <= high)
    exterior = ~interior
    if not interior.any() or not exterior.any():
        raise ValueError("the requested band leaves the interior or exterior empty")

    model, _ = train_surface_model(
        feature_array[interior], targets[interior], config
    )

    # Score the interior on the same held-out split the model early-stopped on,
    # so the interpolation number is not measured on data the weights saw.
    _, validation_index = _split_indices(
        int(interior.sum()), config.validation_fraction, config.seed
    )
    interior_features = feature_array[interior][validation_index]
    interior_targets = targets[interior][validation_index]

    interior_errors = model.predict(interior_features) - interior_targets
    exterior_errors = model.predict(feature_array[exterior]) - targets[exterior]

    return ExtrapolationReport(
        interior_rmse=float(np.sqrt(np.mean(interior_errors**2))),
        exterior_rmse=float(np.sqrt(np.mean(exterior_errors**2))),
        n_interior=int(interior_features.shape[0]),
        n_exterior=int(exterior.sum()),
        band=(float(low), float(high)),
    )
