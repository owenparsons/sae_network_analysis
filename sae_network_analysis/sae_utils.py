# Please note: These functions are taken from one of the SAE Lens tutorials: https://github.com/jbloomAus/SAELens/blob/main/tutorials/tutorial_2_0.ipynb
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, NamedTuple, Callable
from tqdm import tqdm

import torch
from sae_lens import SAE, HookedSAETransformer
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

EPS = 1e-8

class SaeReconstructionCache(NamedTuple):
    sae_in: torch.Tensor
    feature_acts: torch.Tensor
    sae_out: torch.Tensor
    sae_error: torch.Tensor


def track_grad(tensor: torch.Tensor) -> None:
    """wrapper around requires_grad and retain_grad"""
    tensor.requires_grad_(True)
    tensor.retain_grad()


@dataclass
class ApplySaesAndRunOutput:
    model_output: torch.Tensor
    model_activations: dict[str, torch.Tensor]
    sae_activations: dict[str, SaeReconstructionCache]

    def zero_grad(self) -> None:
        """Helper to zero grad all tensors in this object."""
        self.model_output.grad = None
        for act in self.model_activations.values():
            act.grad = None
        for cache in self.sae_activations.values():
            cache.sae_in.grad = None
            cache.feature_acts.grad = None
            cache.sae_out.grad = None
            cache.sae_error.grad = None


def apply_saes_and_run(
    model: HookedTransformer,
    saes: dict[str, SAE],
    input: Any,
    include_error_term: bool = True,
    track_model_hooks: list[str] | None = None,
    return_type: Literal["logits", "loss"] = "logits",
    track_grads: bool = False,
) -> ApplySaesAndRunOutput:
    """
    Apply the SAEs to the model at the specific hook points, and run the model.
    By default, this will include a SAE error term which guarantees that the SAE
    will not affect model output. This function is designed to work correctly with
    backprop as well, so it can be used for gradient-based feature attribution.

    Args:
        model: the model to run
        saes: the SAEs to apply
        input: the input to the model
        include_error_term: whether to include the SAE error term to ensure the SAE doesn't affect model output. Default True
        track_model_hooks: a list of hook points to record the activations and gradients. Default None
        return_type: this is passed to the model.run_with_hooks function. Default "logits"
        track_grads: whether to track gradients. Default False
    """

    fwd_hooks = []
    bwd_hooks = []

    sae_activations: dict[str, SaeReconstructionCache] = {}
    model_activations: dict[str, torch.Tensor] = {}

    # this hook just track the SAE input, output, features, and error. If `track_grads=True`, it also ensures
    # that requires_grad is set to True and retain_grad is called for intermediate values.
    def reconstruction_hook(sae_in: torch.Tensor, hook: HookPoint, hook_point: str):  # noqa: ARG001
        sae = saes[hook_point]
        feature_acts = sae.encode(sae_in)
        sae_out = sae.decode(feature_acts)
        sae_error = (sae_in - sae_out).detach().clone()
        if track_grads:
            track_grad(sae_error)
            track_grad(sae_out)
            track_grad(feature_acts)
            track_grad(sae_in)
        sae_activations[hook_point] = SaeReconstructionCache(
            sae_in=sae_in,
            feature_acts=feature_acts,
            sae_out=sae_out,
            sae_error=sae_error,
        )

        if include_error_term:
            return sae_out + sae_error
        return sae_out

    def sae_bwd_hook(output_grads: torch.Tensor, hook: HookPoint):  # noqa: ARG001
        # this just passes the output grads to the input, so the SAE gets the same grads despite the error term hackery
        return (output_grads,)

    # this hook just records model activations, and ensures that intermediate activations have gradient tracking turned on if needed
    def tracking_hook(hook_input: torch.Tensor, hook: HookPoint, hook_point: str):  # noqa: ARG001
        model_activations[hook_point] = hook_input
        if track_grads:
            track_grad(hook_input)
        return hook_input

    for hook_point in saes.keys():
        fwd_hooks.append(
            (hook_point, partial(reconstruction_hook, hook_point=hook_point))
        )
        bwd_hooks.append((hook_point, sae_bwd_hook))
    for hook_point in track_model_hooks or []:
        fwd_hooks.append((hook_point, partial(tracking_hook, hook_point=hook_point)))

    # now, just run the model while applying the hooks
    with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
        model_output = model(input, return_type=return_type)

    return ApplySaesAndRunOutput(
        model_output=model_output,
        model_activations=model_activations,
        sae_activations=sae_activations,
    )

torch.set_grad_enabled(True)
@dataclass
class AttributionGrads:
    metric: torch.Tensor
    model_output: torch.Tensor
    model_activations: dict[str, torch.Tensor]
    sae_activations: dict[str, SaeReconstructionCache]


@dataclass
class Attribution:
    model_attributions: dict[str, torch.Tensor]
    model_activations: dict[str, torch.Tensor]
    model_grads: dict[str, torch.Tensor]
    sae_feature_attributions: dict[str, torch.Tensor]
    sae_feature_activations: dict[str, torch.Tensor]
    sae_feature_grads: dict[str, torch.Tensor]
    sae_errors_attribution_proportion: dict[str, float]


def calculate_attribution_grads(
    model: HookedSAETransformer,
    prompt: str,
    pos_token: Any,
    neg_token: Any,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    track_hook_points: list[str] | None = None,
    include_saes: dict[str, SAE] | None = None,
    return_logits: bool = True,
    include_error_term: bool = True,
) -> AttributionGrads:
    """
    Wrapper around apply_saes_and_run that calculates gradients wrt to the metric_fn.
    Tracks grads for both SAE feature and model neurons, and returns them in a structured format.
    """
    output = apply_saes_and_run(
        model,
        saes=include_saes or {},
        input=prompt,
        return_type="logits" if return_logits else "loss",
        track_model_hooks=track_hook_points,
        include_error_term=include_error_term,
        track_grads=True,
    )
    metric = metric_fn(output.model_output, pos_token=pos_token, neg_token=neg_token)
    output.zero_grad()
    metric.backward()
    return AttributionGrads(
        metric=metric,
        model_output=output.model_output,
        model_activations=output.model_activations,
        sae_activations=output.sae_activations,
    )


def calculate_feature_attribution(
    model: HookedSAETransformer,
    input: Any,
    pos_token: Any,
    neg_token: Any,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    track_hook_points: list[str] | None = None,
    include_saes: dict[str, SAE] | None = None,
    return_logits: bool = True,
    include_error_term: bool = True,
) -> Attribution:
    """
    Calculate feature attribution for SAE features and model neurons following
    the procedure in https://transformer-circuits.pub/2024/march-update/index.html#feature-heads.
    This include the SAE error term by default, so inserting the SAE into the calculation is
    guaranteed to not affect the model output. This can be disabled by setting `include_error_term=False`.

    Args:
        model: The model to calculate feature attribution for.
        input: The input to the model.
        metric_fn: A function that takes the model output and returns a scalar metric.
        track_hook_points: A list of model hook points to track activations for, if desired
        include_saes: A dictionary of SAEs to include in the calculation. The key is the hook point to apply the SAE to.
        return_logits: Whether to return the model logits or loss. This is passed to TLens, so should match whatever the metric_fn expects (probably logits)
        include_error_term: Whether to include the SAE error term in the calculation. This is recommended, as it ensures that the SAE will not affecting the model output.
    """
    # first, calculate gradients wrt to the metric_fn.
    # these will be multiplied with the activation values to get the attributions
    outputs_with_grads = calculate_attribution_grads(
        model,
        input,
        pos_token,
        neg_token,
        metric_fn,
        track_hook_points,
        include_saes=include_saes,
        return_logits=return_logits,
        include_error_term=include_error_term,
    )
    model_attributions = {}
    model_activations = {}
    model_grads = {}
    sae_feature_attributions = {}
    sae_feature_activations = {}
    sae_feature_grads = {}
    sae_error_proportions = {}
    # this code is long, but all it's doing is multiplying the grads by the activations
    # and recording grads, acts, and attributions in dictionaries to return to the user
    with torch.no_grad():
        for name, act in outputs_with_grads.model_activations.items():
            assert act.grad is not None
            raw_activation = act.detach().clone()
            model_attributions[name] = (act.grad * raw_activation).detach().clone()
            model_activations[name] = raw_activation
            model_grads[name] = act.grad.detach().clone()
        for name, act in outputs_with_grads.sae_activations.items():
            assert act.feature_acts.grad is not None
            assert act.sae_out.grad is not None
            raw_activation = act.feature_acts.detach().clone()
            sae_feature_attributions[name] = (
                (act.feature_acts.grad * raw_activation).detach().clone()
            )
            sae_feature_activations[name] = raw_activation
            sae_feature_grads[name] = act.feature_acts.grad.detach().clone()
            if include_error_term:
                assert act.sae_error.grad is not None
                error_grad_norm = act.sae_error.grad.norm().item()
            else:
                error_grad_norm = 0
            sae_out_norm = act.sae_out.grad.norm().item()
            sae_error_proportions[name] = error_grad_norm / (
                sae_out_norm + error_grad_norm + EPS
            )
        return Attribution(
            model_attributions=model_attributions,
            model_activations=model_activations,
            model_grads=model_grads,
            sae_feature_attributions=sae_feature_attributions,
            sae_feature_activations=sae_feature_activations,
            sae_feature_grads=sae_feature_grads,
            sae_errors_attribution_proportion=sae_error_proportions,
        )

def run_feature_attribution(df, key_column, model, sae, metric_fn):
    """
    Runs the calculate_feature_attribution function on each row of the input DataFrame
    and stores the results in a dictionary using the specified key column.

    :param df: pd.DataFrame, the processed DataFrame with encoded tokens
    :param key_column: str, the column of the input DataFrame to be used as keys for the output dict
    :return: dict, with keys from the specified column and values from calculate_feature_attribution
    """
    # Initialize an empty dictionary to store the results
    attribution_dict = {}

    # Iterate over each row of the DataFrame and calculate feature attribution
    for index, row in df.iterrows():
        # Use the value from the specified key column as the key in the dictionary
        key = row[key_column]
        # Calculate the feature attribution for the current row
        attribution_dict[key] = calculate_feature_attribution(
            input = row['input'],
            pos_token = row['pos_token'],
            neg_token = row['neg_token'],
            model = model,
            metric_fn = metric_fn,
            include_saes={sae.cfg.hook_name: sae},
            include_error_term=True,
            return_logits=True,
        )

    return attribution_dict

def metric_fn(logits: torch.tensor, pos_token: torch.tensor, neg_token: torch.Tensor) -> torch.Tensor:
  return logits[0,-1,pos_token] - logits[0,-1,neg_token]


def find_max_activations(model, sae, activation_store, feature_ids, num_batches=100):
    """
    Find the maximum activation for a set of feature indices.

    This function is designed to identify the maximum activation values for specific
    features across multiple batches. It runs the model with a cached set of activations
    and determines the peak activation for each feature in the given feature set.
    This is particularly useful when fine-tuning or calibrating the strength of
    steering or feature manipulation techniques.

    Parameters:
    ----------
    model : torch.nn.Module
        The model that generates the activations.
    sae : CustomSaeModule
        SAE model used for encoding the input and producing feature activations.
    activation_store : ActivationStore
        An object for storing and retrieving activations.
    feature_ids : torch.Tensor
        A tensor containing the indices of features for which we want to calculate maximum activations.
    num_batches : int, optional
        The number of batches to run through the model to calculate maximum activations.
        Defaults to 100.

    Returns:
    -------
    torch.Tensor
        A tensor containing the maximum activation value for each feature in `feature_ids`.

    """
    max_activation = torch.tensor([0.0]*feature_ids.shape[0], device=feature_ids.device)

    pbar = tqdm(range(num_batches))
    for _ in pbar:
        tokens = activation_store.get_batch_tokens()

        _, cache = model.run_with_cache(
            tokens,
            stop_at_layer=sae.cfg.hook_layer + 1,
            names_filter=[sae.cfg.hook_name]
        )

        sae_in = cache[sae.cfg.hook_name]
        feature_acts = sae.encode(sae_in).squeeze()

        feature_acts = feature_acts.flatten(0, 1)
        batch_max_activation = feature_acts[:, feature_ids].max(0).values

        batch_max_activation = batch_max_activation.to(max_activation.device)
        max_activation = torch.max(max_activation, batch_max_activation)

        pbar.set_description(f"Max activations: {max_activation.tolist()}")

    return max_activation


def steering(activations, hook, steering_strength=1.0, steering_vector=None, max_act=1.0):
    """
    Modify activations using steering vectors.

    This function adjusts the activations of a neural network by applying a steering vector,
    which is designed to nudge the network in a desired direction by adding to the existing
    activation.

    Parameters:
    ----------
    activations : torch.Tensor
        The activations to be modified. Typically a tensor of activations from a specific layer
        in the model during a forward pass.
    hook : torch.nn.Module
        A hook object that is used to modify the forward pass of the model at a specific point.
    steering_strength : float, optional
        A scalar value controlling how much to steer the activations. Defaults to 1.0.
    steering_vector : torch.Tensor, optional
        A tensor representing the vector by which to steer the activations.
    max_act : float, optional
        The maximum activation observed for the given features being steered.

    Returns:
    -------
    torch.Tensor
        The modified activations with steering applied.

    """
    try:
        return activations + torch.sum(max_act.view(-1, 1) * steering_strength * steering_vector, dim=0)
    except Exception as e:
        print(f"Error in steering hook: {e}")
        import pdb; pdb.set_trace()

def generate_with_steering(model, sae, prompt, answer, steering_features, max_act, steering_strength=1.0, max_new_tokens=95, output_token=True):
    """
    Generate text using a model with steering applied to specific features.

    This function modifies the text generation process of a model by steering its activations
    toward specific features. It uses a hook to alter activations during the generation
    process, which allows for controlled steering of the model’s outputs based on a given
    set of features and their maximum activations.

    Parameters:
    ----------
    model : torch.nn.Module
        The language model used for text generation.
    sae : CustomSaeModule
        An instance of a specialized SAE (Sparse Autoencoder) model used for encoding features
        and determining the feature steering vectors.
    prompt : str
        The input text prompt used for generating the text.
    answer : str
        The expected or correct answer to be used for post-generation evaluation.
    steering_features : torch.Tensor
        A tensor containing the indices of the features that will be used for steering during generation.
    max_act : torch.Tensor
        A tensor containing the maximum activations for the steering features. This is used to
        scale the steering strength.
    steering_strength : float, optional
        A scalar controlling how strongly the activations should be steered. Defaults to 1.0.
    max_new_tokens : int, optional
        The maximum number of new tokens to generate in response to the prompt. Defaults to 95.
    output_token : bool, optional
        If True, the generated output will be returned as tokens; otherwise, it returns the decoded text.
        Defaults to True.

    Returns:
    -------
    str or torch.Tensor
        The generated text or the output tokens, depending on the value of `output_token`.
    """
    input_ids = model.to_tokens(prompt, prepend_bos=sae.cfg.prepend_bos)

    steering_vectors = sae.W_dec[steering_features].to(model.cfg.device)

    steering_hook = partial(
        steering,
        steering_vector=steering_vectors,
        steering_strength=steering_strength,
        max_act=max_act
    )

    # standard transformerlens syntax for a hook context for generation
    with model.hooks(fwd_hooks=[(sae.cfg.hook_name, steering_hook)]):
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            stop_at_eos = False if device == "mps" else True,
            prepend_bos = sae.cfg.prepend_bos,
        )

        test_prompt(prompt, answer, model)

    return model.tokenizer.decode(output[0]) if output_token else output
