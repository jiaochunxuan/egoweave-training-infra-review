import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def future_latent_cosine_loss(prediction, target, mask, *, eps=1e-6):
    """Per-sample cosine distance to a frozen future-video representation."""

    prediction = jnp.asarray(prediction)
    target = jnp.asarray(target)
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target must have the same shape, got {prediction.shape} and {target.shape}")
    if mask.shape != prediction.shape[:-1]:
        raise ValueError(f"mask shape {mask.shape} does not match batch shape {prediction.shape[:-1]}")
    prediction = prediction / jnp.maximum(jnp.linalg.norm(prediction, axis=-1, keepdims=True), eps)
    target = target / jnp.maximum(jnp.linalg.norm(target, axis=-1, keepdims=True), eps)
    distance = 1.0 - jnp.sum(prediction * jax.lax.stop_gradient(target), axis=-1)
    return jnp.where(mask, distance, jnp.zeros_like(distance))


def reduce_flow_matching_error(
    error,
    *,
    loss_type="mse",
    reliability=None,
    action_loss_weights=None,
    delta=1.0,
    eps=1e-6,
):
    """Reduce a flow-matching velocity error without changing its batch/horizon axes."""
    error = jnp.asarray(error)
    effective_weight = None
    if action_loss_weights is not None:
        fixed = jnp.asarray(action_loss_weights, dtype=error.dtype)
        expected_shape = (error.shape[-1],)
        if fixed.shape != expected_shape:
            raise ValueError(f"action_loss_weights shape {fixed.shape} does not match {expected_shape}")
        effective_weight = fixed
    if reliability is not None:
        reliability = jnp.asarray(reliability, dtype=error.dtype)
        if reliability.shape != error.shape:
            raise ValueError(f"reliability shape {reliability.shape} does not match error shape {error.shape}")
        effective_weight = reliability if effective_weight is None else reliability * effective_weight

    def _weighted_reduce(per_dim_loss):
        if effective_weight is None:
            return jnp.mean(per_dim_loss, axis=-1)
        weight_sum = jnp.sum(effective_weight, axis=-1)
        weighted_sum = jnp.sum(per_dim_loss * effective_weight, axis=-1)
        # Avoid division by zero in both primal and derivative graphs. Empty
        # supervision rows therefore have exact zero loss and error gradient.
        safe_weight_sum = jnp.where(weight_sum > 0, weight_sum, jnp.ones_like(weight_sum))
        reduced = weighted_sum / safe_weight_sum
        return jnp.where(weight_sum > 0, reduced, jnp.zeros_like(reduced))

    if loss_type == "mse":
        return _weighted_reduce(jnp.square(error))
    magnitude = jnp.abs(error)
    huber = jnp.where(magnitude <= delta, 0.5 * jnp.square(error), delta * (magnitude - 0.5 * delta))
    if loss_type == "huber":
        return _weighted_reduce(huber)
    if loss_type == "dynamic_masked_mse":
        if reliability is None:
            raise ValueError("reliability is required for dynamic_masked_mse")
        return _weighted_reduce(jnp.square(error))
    if loss_type != "reliability_huber":
        raise ValueError(f"Unknown flow-matching loss type: {loss_type}")
    if reliability is None:
        raise ValueError("reliability is required for reliability_huber")
    return _weighted_reduce(huber)


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.image_keys = config.image_keys
        self.preserve_image_token_layout = config.preserve_image_token_layout
        self.image_token_layout = _model.IMAGE_KEYS if config.preserve_image_token_layout else config.image_keys
        self.action_loss_weights = config.action_loss_weights
        self.flow_loss_type = config.flow_loss_type
        self.human_action_loss_mode = config.human_action_loss_mode
        self.reliability_weighted_huber = config.reliability_weighted_huber
        self.reliability_huber_delta = config.reliability_huber_delta
        self.quest_local21_recon_loss_weight = config.quest_local21_recon_loss_weight
        self.quest_local21_recon_left_weight = config.quest_local21_recon_left_weight
        self.quest_local21_recon_right_weight = config.quest_local21_recon_right_weight
        self.quest_local21_recon_left_scale = config.quest_local21_recon_left_scale
        self.quest_local21_recon_right_scale = config.quest_local21_recon_right_scale
        self.quest_local21_recon_wrist_position_absolute = config.quest_local21_recon_wrist_position_absolute
        self.quest_local21_recon_wrist_rotation_absolute = config.quest_local21_recon_wrist_rotation_absolute
        self.quest_local21_recon_wrist_pose_relative_to_head = config.quest_local21_recon_wrist_pose_relative_to_head
        self.quest_local21_recon_state_mean = config.quest_local21_recon_state_mean
        self.quest_local21_recon_state_std = config.quest_local21_recon_state_std
        self.quest_local21_recon_action_mean = config.quest_local21_recon_action_mean
        self.quest_local21_recon_action_std = config.quest_local21_recon_action_std
        self.future_latent_dim = config.future_latent_dim
        self.future_latent_loss_weight = config.future_latent_loss_weight
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # MoT: optionally create a second action expert for ego data.
        self.mot_enabled = config.ego_action_expert_variant is not None
        if self.mot_enabled:
            ego_action_expert_config = _gemma.get_config(config.ego_action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config] + (
                    [ego_action_expert_config] if self.mot_enabled else []
                ),
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=(
            [False, True, True] if (config.pi05 and self.mot_enabled)
            else [False, True] if config.pi05
            else [False, False, False] if self.mot_enabled
            else [False, False]
        ))
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.state_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Canonical Interaction Bridge (CIB).  Only the final projection is
        # zero-initialized so the new path starts exactly equivalent to the
        # original Pi0.5 while still receiving gradients on the first step.
        self.cib_enabled = config.cib_enabled
        if self.cib_enabled:
            self.cib_input_dim = config.cib_input_dim
            self.cib_state_proj_in = nnx.Linear(
                self.cib_input_dim, config.cib_hidden_dim, rngs=rngs
            )
            self.cib_state_proj_mid = nnx.Linear(
                config.cib_hidden_dim, config.cib_hidden_dim, rngs=rngs
            )
            self.cib_bridge_out = nnx.Linear(
                config.cib_hidden_dim,
                action_expert_config.width,
                rngs=rngs,
                kernel_init=nnx.initializers.zeros,
                bias_init=nnx.initializers.zeros,
            )

        # Source conditioning: learnable embedding for data source (0=null, 1=ego, 2=robot).
        self.source_conditioning = config.source_conditioning
        self.source_dropout_rate = config.source_dropout_rate
        if self.source_conditioning:
            self.source_embed = nnx.Embed(
                num_embeddings=3,
                # The source token joins the prefix stream (PaliGemma width),
                # NOT the action-expert stream. Using the expert width here
                # breaks embed_prefix's jnp.concatenate.
                features=_gemma.get_config(config.paligemma_variant).width,
                rngs=rngs,
            )
        # Default source type (0=null, 1=ego, 2=robot) used when source_id is not passed.
        self.default_source_type = config.default_source_type

        # Progress prediction head: predicts normalised task progress from
        # the pooled prefix representation of the main Gemma model.
        self.progress_loss_weight = config.progress_loss_weight
        if self.progress_loss_weight > 0:
            paligemma_width = _gemma.get_config(config.paligemma_variant).width
            self.progress_mlp_in = nnx.Linear(
                paligemma_width, action_expert_config.width, rngs=rngs
            )
            self.progress_mlp_out = nnx.Linear(
                action_expert_config.width, 1, rngs=rngs
            )

        if self.future_latent_dim > 0:
            paligemma_width = _gemma.get_config(config.paligemma_variant).width
            self.future_latent_predictor_in = nnx.Linear(
                paligemma_width, action_expert_config.width, rngs=rngs
            )
            self.future_latent_predictor_out = nnx.Linear(
                action_expert_config.width, self.future_latent_dim, rngs=rngs
            )

        # MoT: separate output projection for ego action expert.
        if self.mot_enabled:
            self.ego_action_out_proj = nnx.Linear(
                action_expert_config.width, config.action_dim, rngs=rngs
            )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation, *, source_id: at.Int[at.Array, " b"] | None = None
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        if self.preserve_image_token_layout:
            # Encode only selected real views, retaining false-masked slots so
            # downstream dense attention still uses the original shapes.
            encoded_images = {
                name: self.PaliGemma.img(obs.images[name], train=False)[0] for name in self.image_keys
            }
            template_tokens = next(iter(encoded_images.values()))
            for name in self.image_token_layout:
                if name in encoded_images:
                    image_tokens = encoded_images[name]
                    image_mask = obs.image_masks[name]
                else:
                    image_tokens = jnp.zeros_like(template_tokens)
                    image_mask = jnp.zeros(template_tokens.shape[:1], dtype=jnp.bool_)
                tokens.append(image_tokens)
                input_mask.append(einops.repeat(image_mask, "b -> b s", s=image_tokens.shape[1]))
                ar_mask += [False] * image_tokens.shape[1]
        else:
            # Original image-embedding branch, unchanged for default policies.
            for name in obs.images:
                image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

                tokens.append(image_tokens)
                input_mask.append(
                    einops.repeat(
                        obs.image_masks[name],
                        "b -> b s",
                        s=image_tokens.shape[1],
                    )
                )
                # image tokens attend to each other
                ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        # Append source token if source conditioning is enabled.
        if self.source_conditioning and source_id is not None:
            source_token = self.source_embed(source_id)  # (b, width)
            source_token = source_token[:, None, :]  # (b, 1, width)
            tokens.append(source_token)
            input_mask.append(jnp.ones((source_token.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [False]  # full attention to image/language

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _apply_cib(
        self, suffix_out: at.Float[at.Array, "b s emb"], observation: _model.Observation
    ) -> at.Float[at.Array, "b s emb"]:
        """Add a CIB residual from the current normalized state descriptor."""
        if not self.cib_enabled:
            return suffix_out

        state = jnp.asarray(observation.state, dtype=suffix_out.dtype)
        # Model transforms may expose either a human state (156D) or a robot
        # state (23D).  The bridge interface is deliberately fixed-width; the
        # upstream dataset is responsible for placing its canonical descriptor
        # in the leading slots and unused slots are zero-padded here.
        state_width = state.shape[-1]
        if state_width >= self.cib_input_dim:
            descriptor = state[..., : self.cib_input_dim]
        else:
            descriptor = jnp.pad(
                state,
                [(0, 0)] * (state.ndim - 1) + [(0, self.cib_input_dim - state_width)],
            )
        descriptor = jnp.nan_to_num(descriptor)
        hidden = nnx.swish(self.cib_state_proj_in(descriptor))
        hidden = nnx.swish(self.cib_state_proj_mid(hidden))
        residual = self.cib_bridge_out(hidden)
        return suffix_out + residual[:, None, :]

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False,
        source_id: at.Int[at.Array, " b"] | None = None,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train, image_keys=self.image_keys
        )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Use default source type if source_id not provided.
        if self.source_conditioning and source_id is None:
            source_id = jnp.full(
                (observation.state.shape[0],), self.default_source_type, dtype=jnp.int32
            )
        # Apply source dropout during training.
        if self.source_conditioning and train and self.source_dropout_rate > 0 and source_id is not None:
            rng, sub_rng = jax.random.split(rng)
            dropout_mask = jax.random.bernoulli(
                sub_rng, self.source_dropout_rate, (source_id.shape[0],)
            )
            source_id = jnp.where(dropout_mask, 0, source_id)

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
            observation, source_id=source_id
        )
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        # MoT: route suffix to robot (expert 1) or ego (expert 2) expert.
        if self.mot_enabled and source_id is not None:
            # STATIC routing (jit-safe): JAX forbids converting a traced array
            # to Python bool (TracerBoolConversionError). Route by the run's
            # default source type: ego pretrain (--source-type 1) -> expert 2,
            # robot finetune (--source-type 2) -> expert 1. Per-sample source
            # values still flow into the source embedding (jit-safe).
            use_ego = self.default_source_type == 1
        else:
            use_ego = False

        if self.mot_enabled and use_ego:
            llm_out, _ = self.PaliGemma.llm(
                [prefix_tokens, None, suffix_tokens],
                mask=attn_mask, positions=positions,
                adarms_cond=[None, None, adarms_cond],
            )
            robot_out, ego_out = llm_out[1], llm_out[2]
            suffix_out = self._apply_cib(ego_out, observation)
            v_t = self.ego_action_out_proj(suffix_out[:, -self.action_horizon :])
        else:
            llm_out, _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens] + ([None] if self.mot_enabled else []),
                mask=attn_mask, positions=positions,
                adarms_cond=[None, adarms_cond] + ([None] if self.mot_enabled else []),
            )
            suffix_out = self._apply_cib(llm_out[1], observation)
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # --- Auxiliary progress-prediction loss ---
        if self.progress_loss_weight > 0 and observation.progress is not None:
            # Mean-pool valid prefix tokens (mask out padding).
            pmask = prefix_mask[:, :, None]  # (b, s, 1)
            prefix_pool = (llm_out[0] * pmask).sum(axis=1) / pmask.sum(axis=1).clip(min=1)
            progress_pred = self.progress_mlp_out(
                jax.nn.relu(self.progress_mlp_in(prefix_pool))
            ).squeeze(-1)  # (b,)
            progress_pred = jax.nn.sigmoid(progress_pred)
            progress_loss = jnp.mean((progress_pred - observation.progress) ** 2)
        else:
            progress_loss = 0.0

        if self.future_latent_loss_weight > 0:
            if observation.future_latent is None or observation.future_latent_mask is None:
                raise ValueError("future_latent and future_latent_mask are required when future latent loss is enabled")
            pmask = prefix_mask[:, :, None]
            prefix_pool = (llm_out[0] * pmask).sum(axis=1) / pmask.sum(axis=1).clip(min=1)
            future_prediction = self.future_latent_predictor_out(
                nnx.swish(self.future_latent_predictor_in(prefix_pool))
            )
            future_loss = future_latent_cosine_loss(
                future_prediction,
                observation.future_latent,
                observation.future_latent_mask,
            )
        else:
            future_loss = jnp.zeros(batch_shape, dtype=v_t.dtype)

        error = v_t - u_t
        if observation.reliability is not None and self.human_action_loss_mode is None and not self.reliability_weighted_huber:
            # Boolean label availability, never an input feature/future oracle.
            chunked_loss = reduce_flow_matching_error(
                error, loss_type=self.flow_loss_type,
                reliability=observation.reliability,
                action_loss_weights=self.action_loss_weights,
            )
            available = jnp.any(observation.reliability > 0, axis=-1)
            chunked_loss *= available.shape[-1] / jnp.maximum(available.sum(axis=-1, keepdims=True), 1)
        elif self.human_action_loss_mode is not None:
            if observation.reliability is None:
                raise ValueError("reliability sidecar is required for dynamic human-action MSE")
            if self.quest_local21_recon_loss_weight != 0.0:
                raise ValueError(
                    "QUEST local21 reconstruction loss weight must be exactly 0 when dynamic reliability is present"
                )
            chunked_loss = reduce_flow_matching_error(
                error,
                loss_type="dynamic_masked_mse",
                reliability=observation.reliability,
                action_loss_weights=self.action_loss_weights,
            )
        elif self.reliability_weighted_huber:
            if observation.reliability is None:
                raise ValueError("reliability sidecar is required when reliability_weighted_huber=True")
            chunked_loss = reduce_flow_matching_error(
                error,
                loss_type="reliability_huber",
                reliability=observation.reliability,
                action_loss_weights=self.action_loss_weights,
                delta=self.reliability_huber_delta,
            )
        else:
            chunked_loss = reduce_flow_matching_error(
                error,
                loss_type=self.flow_loss_type,
                action_loss_weights=self.action_loss_weights,
                delta=self.reliability_huber_delta,
            )

        total_loss = (
            chunked_loss
            + self.progress_loss_weight * progress_loss
            + self.future_latent_loss_weight * future_loss[..., None]
        )
        if self.quest_local21_recon_loss_weight <= 0.0:
            return total_loss

        # Flow matching predicts velocity u_t = noise - action. Convert the
        # predicted velocity at x_t back to the implied action x_0 before
        # applying the QUEST-specific geometric loss.
        pred_actions = x_t - time_expanded * v_t
        recon_loss = self._quest_local21_reconstruction_loss(
            observation.state, pred_actions, actions, reliability=observation.reliability)
        return total_loss + self.quest_local21_recon_loss_weight * recon_loss

    def _quest_local21_reconstruction_loss(
        self,
        state: at.Float[at.Array, "b ad"],
        pred_actions: at.Float[at.Array, "b ah ad"],
        target_actions: at.Float[at.Array, "b ah ad"],
        reliability=None,
    ) -> at.Float[at.Array, "b ah"]:
        state_mean = jnp.asarray(self.quest_local21_recon_state_mean, dtype=state.dtype)
        state_std = jnp.asarray(self.quest_local21_recon_state_std, dtype=state.dtype)
        action_mean = jnp.asarray(self.quest_local21_recon_action_mean, dtype=pred_actions.dtype)
        action_std = jnp.asarray(self.quest_local21_recon_action_std, dtype=pred_actions.dtype)

        raw_state = state * (state_std + 1e-6) + state_mean
        raw_pred = pred_actions * (action_std + 1e-6) + action_mean
        raw_target = target_actions * (action_std + 1e-6) + action_mean
        raw_state = raw_state[:, : raw_pred.shape[-1]]

        left_loss = self._quest_local21_side_reconstruction_loss(
            raw_state,
            raw_pred,
            raw_target,
            wrist_pos_slice=slice(9, 12),
            wrist_rot_slice=slice(12, 18),
            local_slice=slice(27, 90),
            hand_scale=self.quest_local21_recon_left_scale,
        )
        right_loss = self._quest_local21_side_reconstruction_loss(
            raw_state,
            raw_pred,
            raw_target,
            wrist_pos_slice=slice(18, 21),
            wrist_rot_slice=slice(21, 27),
            local_slice=slice(90, 153),
            hand_scale=self.quest_local21_recon_right_scale,
        )
        if reliability is not None:
            head_ok = jnp.all(reliability[..., :9] > 0, axis=-1)
            left_ok = head_ok & jnp.all(reliability[..., 9:18] > 0, axis=-1) & jnp.all(reliability[..., 27:90] > 0, axis=-1)
            right_ok = head_ok & jnp.all(reliability[..., 18:27] > 0, axis=-1) & jnp.all(reliability[..., 90:153] > 0, axis=-1)
            lw = left_ok * self.quest_local21_recon_left_weight
            rw = right_ok * self.quest_local21_recon_right_weight
            denom = lw + rw
            result = (jnp.where(left_ok, left_loss, 0) * lw + jnp.where(right_ok, right_loss, 0) * rw) / jnp.maximum(denom, 1e-6)
            available = denom > 0
            return result * available.shape[-1] / jnp.maximum(available.sum(axis=-1, keepdims=True), 1)
        denom = self.quest_local21_recon_left_weight + self.quest_local21_recon_right_weight
        denom = max(denom, 1e-6)
        return (
            self.quest_local21_recon_left_weight * left_loss
            + self.quest_local21_recon_right_weight * right_loss
        ) / denom

    def _quest_local21_side_reconstruction_loss(
        self,
        raw_state: at.Float[at.Array, "b ad"],
        raw_pred: at.Float[at.Array, "b ah ad"],
        raw_target: at.Float[at.Array, "b ah ad"],
        *,
        wrist_pos_slice: slice,
        wrist_rot_slice: slice,
        local_slice: slice,
        hand_scale: float,
    ) -> at.Float[at.Array, "b ah"]:
        current_wrist_pos = raw_state[:, wrist_pos_slice]
        current_wrist_rot = self._rotation_6d_to_matrix(raw_state[:, wrist_rot_slice])
        if self.quest_local21_recon_wrist_pose_relative_to_head:
            current_head_pos = raw_state[:, 0:3]
            current_head_rot = self._rotation_6d_to_matrix(raw_state[:, 3:9])
            pred_head_pos = current_head_pos[:, None, :] + raw_pred[:, :, 0:3]
            target_head_pos = current_head_pos[:, None, :] + raw_target[:, :, 0:3]
            pred_head_rel_rot = self._rotation_6d_to_matrix(raw_pred[:, :, 3:9])
            target_head_rel_rot = self._rotation_6d_to_matrix(raw_target[:, :, 3:9])
            pred_head_rot = jnp.einsum("bij,bhjk->bhik", current_head_rot, pred_head_rel_rot)
            target_head_rot = jnp.einsum("bij,bhjk->bhik", current_head_rot, target_head_rel_rot)
            pred_wrist_pos = pred_head_pos + jnp.einsum(
                "bhij,bhj->bhi",
                pred_head_rot,
                raw_pred[:, :, wrist_pos_slice],
            )
            target_wrist_pos = target_head_pos + jnp.einsum(
                "bhij,bhj->bhi",
                target_head_rot,
                raw_target[:, :, wrist_pos_slice],
            )
        elif self.quest_local21_recon_wrist_position_absolute:
            pred_wrist_pos = raw_pred[:, :, wrist_pos_slice]
            target_wrist_pos = raw_target[:, :, wrist_pos_slice]
        else:
            pred_wrist_pos = current_wrist_pos[:, None, :] + raw_pred[:, :, wrist_pos_slice]
            target_wrist_pos = current_wrist_pos[:, None, :] + raw_target[:, :, wrist_pos_slice]

        pred_action_rot = self._rotation_6d_to_matrix(raw_pred[:, :, wrist_rot_slice])
        target_action_rot = self._rotation_6d_to_matrix(raw_target[:, :, wrist_rot_slice])
        if self.quest_local21_recon_wrist_pose_relative_to_head:
            pred_wrist_rot = jnp.einsum("bhij,bhjk->bhik", pred_head_rot, pred_action_rot)
            target_wrist_rot = jnp.einsum("bhij,bhjk->bhik", target_head_rot, target_action_rot)
        elif self.quest_local21_recon_wrist_rotation_absolute:
            pred_wrist_rot = pred_action_rot
            target_wrist_rot = target_action_rot
        else:
            pred_wrist_rot = jnp.einsum("bij,bhjk->bhik", current_wrist_rot, pred_action_rot)
            target_wrist_rot = jnp.einsum("bij,bhjk->bhik", current_wrist_rot, target_action_rot)

        pred_local = raw_pred[:, :, local_slice].reshape(raw_pred.shape[0], raw_pred.shape[1], 21, 3) * hand_scale
        target_local = (
            raw_target[:, :, local_slice].reshape(raw_target.shape[0], raw_target.shape[1], 21, 3) * hand_scale
        )
        pred_world = pred_wrist_pos[:, :, None, :] + jnp.einsum("bhij,bhnj->bhni", pred_wrist_rot, pred_local)
        target_world = target_wrist_pos[:, :, None, :] + jnp.einsum(
            "bhij,bhnj->bhni",
            target_wrist_rot,
            target_local,
        )
        return jnp.mean(jnp.square(pred_world - target_world), axis=(-1, -2))

    @staticmethod
    def _rotation_6d_to_matrix(rot_6d: at.Float[at.Array, "... 6"]) -> at.Float[at.Array, "... 3 3"]:
        a1 = rot_6d[..., 0:3]
        a2 = rot_6d[..., 3:6]
        b1 = a1 / jnp.maximum(jnp.linalg.norm(a1, axis=-1, keepdims=True), 1e-6)
        b2 = a2 - jnp.sum(a2 * b1, axis=-1, keepdims=True) * b1
        b2 = b2 / jnp.maximum(jnp.linalg.norm(b2, axis=-1, keepdims=True), 1e-6)
        b3 = jnp.cross(b1, b2, axis=-1)
        return jnp.stack([b1, b2, b3], axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=self.image_keys
        )
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None] + ([None] if self.mot_enabled else []),
            mask=prefix_attn_mask, positions=positions,
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            llm_out, _ = self.PaliGemma.llm(
                [None, suffix_tokens] + ([None] if self.mot_enabled else []),
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond] + ([None] if self.mot_enabled else []),
            )
            suffix_out = self._apply_cib(llm_out[1], observation)
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
