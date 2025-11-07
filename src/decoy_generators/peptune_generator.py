import collections
import os
import torch
import torch

from hydra import compose,  initialize
from random import Random
from typing import List, Optional
from transformers import PreTrainedTokenizer
from SmilesPE.tokenizer import SPE_Tokenizer

from src.decoy_generators.ml_generator import MlGenerator, MlGeneratorType

class PeptuneGenerator(MlGenerator):
    def __init__(
            self,
            local_path: str,
            random: Random,
            special_amino_acids: List[str],
            mask_percent: float = 0.3,  # should be between 0.0 and 1.0
            sort_optimization: bool = True,
            batch_size: int = 64,
            ml_generator_type: MlGeneratorType = MlGeneratorType.BEST
    ):
        LOCAL_PATH = "models/Peptune"
        with initialize(version_base=None, config_path=LOCAL_PATH):
            config = compose(config_name="config")
        MlGenerator.__init__(self, local_path, random, special_amino_acids, mask_percent, sort_optimization,
                             batch_size, ml_generator_type)
        self.tokenizer = SMILES_SPE_Tokenizer(f"{LOCAL_PATH}/new_vocab.txt", f"{LOCAL_PATH}/new_splits.txt")
        self.model = Diffusion.load_from_checkpoint(
            f"{LOCAL_PATH}/peptune-pretrained.ckpt",
            tokenizer=self.tokenizer,
            config=config,
            map_location="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.eval()

    def __str__(self):
        return f"Peptune.[{self.mask_percent}]"
    
class SMILES_SPE_Tokenizer(PreTrainedTokenizer):
    r"""
    Constructs a SMILES tokenizer. Based on SMILES Pair Encoding (https://github.com/XinhaoLi74/SmilesPE).
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`string`):
            File containing the vocabulary.
        spe_file (:obj:`string`):
            File containing the trained SMILES Pair Encoding vocabulary.
        unk_token (:obj:`string`, `optional`, defaults to "[UNK]"):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`string`, `optional`, defaults to "[SEP]"):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences
            for sequence classification or for a text and a question for question answering.
            It is also used as the last token of a sequence built with special tokens.
        pad_token (:obj:`string`, `optional`, defaults to "[PAD]"):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`string`, `optional`, defaults to "[CLS]"):
            The classifier token which is used when doing sequence classification (classification of the whole
            sequence instead of per-token classification). It is the first token of the sequence when built with
            special tokens.
        mask_token (:obj:`string`, `optional`, defaults to "[MASK]"):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
    """

    def __init__(self, vocab_file, spe_file,
                unk_token="[UNK]",
                sep_token="[SEP]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                mask_token="[MASK]",
                **kwargs):
        if not os.path.isfile(vocab_file):
            raise ValueError("Can't find a vocabulary file at path '{}'.".format(vocab_file))
        if not os.path.isfile(spe_file):
            raise ValueError("Can't find a SPE vocabulary file at path '{}'.".format(spe_file))

        self.vocab = load_vocab(vocab_file)
        self.spe_vocab = open(spe_file, 'r', encoding='utf-8')
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.spe_tokenizer = SPE_Tokenizer(self.spe_vocab)

        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        return self.spe_tokenizer.tokenize(text).split(' ')

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))
    
    # changed encode and decode functions
    def encode(self, token_array):
        token_ids = []
        token_ids.append(2)
        for token in token_array:
            id = self._convert_token_to_id(token)
            token_ids.append(id)
        token_ids.append(3)
        token_ids = torch.tensor([token_ids])
        attn_mask = torch.ones_like(token_ids)
        return {'input_ids': token_ids, 'attention_mask': attn_mask}
    
    def decode(self, token_ids, skip_special_tokens=True):
        token_ids = token_ids.squeeze(0).cpu().tolist()
        token_array = []
        for idx in token_ids:
            if idx == 3:  # Stop decoding when token ID 3 is encountered
                break
            if skip_special_tokens and idx in self.all_special_ids:
                continue  
            token = self._convert_id_to_token(idx)
            token_array.append(token)
        sequence = "".join(token_array)
        return sequence
    
    def batch_decode(self, batch_token_ids, skip_special_tokens=True):
        sequences = []
        for token_ids in batch_token_ids:
            sequences.append(self.decode(token_ids))
        return sequences
    
    def get_token_split(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        
        token_array = []
        for seq_ids in token_ids:
            seq_array = []
            for id in seq_ids:
                token = self._convert_id_to_token(id)
                seq_array.append(token)
            token_array.append(seq_array)
            
        return token_array
    
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True if the token list is already formatted with special tokens for the model
        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:
        ::
            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |
        if token_ids_1 is None, only returns the first portion of the mask (0's).
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, vocab_path):
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.
        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.
        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

class Diffusion(L.LightningModule):
    def __init__(self, config, tokenizer):
        
        super().__init__()
        self.config = config
        #self.save_hyperparameters()
        
        # PeptideCLM tokenizer 
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.mask_token_id = self.tokenizer.mask_token_id
        self.sampler = self.config.sampling.predictor
        self.analyzer = PeptideAnalyzer()
        
        # backbone LM PeptideCLM model
        if self.config.backbone == 'peptideclm':
            self.backbone = peptideclm.EncoderWrapper(self.tokenizer)
            self.backbone.unfreeze_all_layers()
            self.backbone = torch.compile(self.backbone)
        elif self.config.backbone == 'helmgpt':
            self.backbone = helmgpt.GPT(self.config, self.tokenizer)
            #self.backbone = torch.compile(self.backbone)
        elif self.config.backbone == 'roformer':
            self.backbone = roformer.Roformer(self.config, self.tokenizer)
            self.backbone.unfreeze_all_layers()
        elif self.config.backbone == 'finetune_roformer':
            self.backbone = roformer.Roformer(self.config, self.tokenizer)
            self.backbone.freeze_model()
            self.backbone.unfreeze_n_layers(n=8)
        else: 
            Exception('invalid backbone config')
        
        self.neg_infinity = -1000000.0
        self.T = config.T
        # noise schedule for non-peptide bond tokens (default to log-linear)
        self.noise = noise_schedule.get_noise(config)
        # noise schedule for peptide bonds (log-polynomial)
        self.bond_noise = noise_schedule.LogPolyNoise()
        self.time_conditioning = self.config.time_conditioning
        self.fast_forward_epochs = None
        self.fast_forward_batches = None
        
        self.gen_ppl_eval_model_name_or_path = self.config.eval.gen_ppl_eval_model_name_or_path
        self.gen_ppl_metric = Perplexity()
                
        self.lr = self.config.optim.lr
        self.sampling_eps = self.config.training.sampling_eps
        
        metrics = torchmetrics.MetricCollection({
            'nll': NLL(),
            'bpd': BPD(),
            'ppl': Perplexity(),
        })
        metrics.set_dtype(torch.float64)
        self.train_metrics = metrics.clone(prefix='trainer/')
        self.valid_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')
        
        
    """LOSS FOR INVALID PEPTIDES"""
    
    @torch.no_grad()
    def conditional_gumbel(self, logits, D, k):
        """
        Outputs k samples of Q = StandardGumbel(), such that argmax(logits
        + Q) is given by D (one-hot vector).
        Input:
        - logits: Tensor of shape (batch_size, seq_len, vocab_size)
        - D: One-hot tensor of shape (batch_size, seq_len, vocab_size)
        - k: Number of Gumbel samples
        Output:
        - Adjusted logits with shape (k, batch_size, seq_len, vocab_size)
        """

        # iid. exponential samples of shape (k, batch_size, seq_len, vocab_size)
        E = torch.distributions.exponential.Exponential(rate=torch.ones_like(logits)).sample([k])

        # E of the chosen class, shape (k, batch_size, seq_len, 1)
        Ei = (D * E).sum(dim=-1, keepdim=True)

        # Partition function (normalization constant), shape (batch_size, seq_len, 1)
        Z = logits.exp().sum(dim=-1, keepdim=True)

        # Adjusted logits for Gumbel distribution
        adjusted = (
            D * (-torch.log(Ei) + torch.log(Z)) +
            (1 - D) * -torch.log(E / logits.exp() + Ei / Z)
        )

        # Adjusted logits shape: (k, batch_size, seq_len, vocab_size)
        return adjusted - logits
        
    def replace_gradient(self, value, surrogate):
        """
        Returns `value` but backpropagates gradients through `surrogate`.
        """
        return surrogate + (value - surrogate).detach()
    
    def gumbel_rao(self, logits, k, temp=1.0, I=None):
        """
        Returns a categorical sample from logits (over axis=-1) as a
        one-hot vector, with gumbel-rao gradient.
        Input:
        - logits: Tensor of shape (batch_size, seq_len, vocab_size)
        - k: Number of Gumbel samples for Rao-Blackwellization
        - temp: Temperature for softmax
        - I: Optional, precomputed categorical sample tensor of shape (batch_size, seq_len)
        Output:
        - One-hot tensor of shape (batch_size, seq_len, vocab_size)
        with Gumbel-Rao gradient.
        """
        assert logits.shape[-1] == self.tokenizer.vocab_size
        vocab_size = logits.shape[-1]

        if I is None:
            # Sample indices for each token in the batch
            I = torch.distributions.categorical.Categorical(logits=logits).sample()  # (batch_size, seq_len)

        # Convert indices to one-hot encodings, shape (batch_size, seq_len, vocab_size)
        D = torch.nn.functional.one_hot(I, num_classes=vocab_size).float()

        # Generate k different adjusted logits that all evaluate to the same sequence
        adjusted = logits + self.conditional_gumbel(logits, D, k=k)  # (k, batch_size, seq_len, vocab_size)

        # Compute the surrogate by averaging softmax across k samples
        surrogate = torch.nn.functional.softmax(adjusted / temp, dim=-1).mean(dim=0)  # (batch_size, seq_len, vocab_size)

        # Return one-hot representation with surrogate gradient
        return self.replace_gradient(D, surrogate)
    
    def compute_invalid_loss(self, logits, k=None, temp=None):
        """
        Penalizes logits that produce invalid sequences using the `is_peptide` function,
        scaling penalties inversely with token probabilities.
        Args:
            logits: Tensor of shape [batch_size, seq_len, vocab_size].
            k: Number of samples for Gumbel-Rao.
            temp: Temperature for softmax.
        Returns:
            loss: A scalar tensor representing the total loss for invalid sequences.
        """

        #samples = self.gumbel_rao(logits, k=k, temp=temp)  # (batch_size, seq_len, vocab_size)

        # Convert logits to sequences using the tokenizer
        batch_token_ids = logits.argmax(dim=-1).to(self.device)  # (batch_size, seq_len)
        sampled_sequences = self.tokenizer.batch_decode(batch_token_ids)

        # Check validity of each sampled sequence (not differentiable)
        penalties = torch.tensor(
            [1 if not self.analyzer.is_peptide(seq) else 0 for seq in sampled_sequences],
            dtype=torch.float32,
            device=self.device
        )  
        #print(penalties)

        # Compute probabilities for each token (batch_size, seq_length)
        sampled_probs = torch.softmax(logits, dim=-1).gather(dim=-1, index=batch_token_ids.unsqueeze(-1)).squeeze(-1).to(self.device)

        # scale penalties by softmax probability of sampled tokens
        scaled_penalty = penalties[:, None] * sampled_probs # (batch_size, seq_length)
        
        return scaled_penalty.to(self.device)
    
    """DIFFUSION LOSS"""
    
    def sample_t(self, n, device):
        """
            Sample random time steps for batch training
        """
        # sample values uniformly at random from [0, 1)
        eps_t = torch.rand(n, device=device) 
        # antithetic sampling: reduce variance by pairing each sample with complementary sample
        if self.config.training.antithetic_sampling:
            # compute interval between sampled time steps
            offset = torch.arange(n, device=device) / n
            # ensure that each eps value is evenly spaced between [0, 1)
            eps_t = ((eps_t / n) + offset) % 1

        # ensures values are not exactly 0 or 1
        t = (1 - self.config.training.sampling_eps) * eps_t + self.config.training.sampling_eps
        
        return t
    
    def q_xt(self, x, mask_prob):
        """Computes the noisy sample xt.
        Args:
        x: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input. 
        move_chance: float torch.Tensor with shape (batch_size, 1).
        """

        actual_seq_length = (x != 0).sum(dim=-1, keepdim=True)
        #print(actual_seq_length)

        max_mask_length = (actual_seq_length * 0.75).long()

        mask_indices = torch.rand(*x.shape, device=x.device) < mask_prob
        
        restricted_move_indices = torch.zeros_like(mask_indices, dtype=torch.bool)

        for i in range(x.shape[0]):
            true_positions = torch.where(mask_indices[i])[0]
            if len(true_positions) > max_mask_length[i]:
                selected_positions = true_positions[:max_mask_length[i].item()]
                restricted_move_indices[i, selected_positions] = True
            else:
                restricted_move_indices[i] = mask_indices[i]
                
        xt = torch.where(restricted_move_indices, self.tokenizer.mask_token_id, x)

        return xt

    
    def sample_prior(self, *batch_dims):
        """
            Returns array of fully masked sequences with same shape as input
        """
        return self.mask_token_id * torch.ones(* batch_dims, dtype=torch.int64)
    

    """COMPUTING LOSS"""
    
    def compute_diffusion_loss(self, model_output, xt, x0, t):
        """
        Computes diffusion loss term in ELBO 
        (evaluates how accurately the model predicts the token probabilities at each time step)
        
        Inputs:
        - model_output: [sequence length, vocab size, vocab size] array of logits for each token at each sequence position
        - zt: corrupted version of original input x0 at timestep t
        - x0: original input sequence
        - t: timestep
        """
        # compute interval between each timestep
        dt = 1 / self.T
        
        # compute vectorized alpha scaling terms for the logits at timestep s and t
        alpha_t = 1 - t + torch.zeros_like(x0)
        # s = t - dt
        alpha_s = 1 - (t - dt) + torch.zeros_like(x0)
        
        # gather vector of log-probabilities for each token in x0
        # log<x_theta, x>
        log_x_theta_at_x0 = torch.gather(model_output, -1, x0[:, :, None]) # shape (B, L, vocab_size)
        # gather log-probabillities for assigning a masked token at each position in the sequence at time t 
        # log<x_theta, m>
        log_x_theta_at_m = model_output[:, :, self.mask_token_id]
        # obtain non-log probability of assigning a masked token
        # <xt, m>
        x_theta_at_m = log_x_theta_at_m.exp()
        
        # first term of diffusion loss
        term_1_coef = dt / t
        term_1_log_numerator = torch.log((alpha_t * x_theta_at_m) / t + 1)
        term_1_log_denom = log_x_theta_at_x0
        
        # second term of diffusion loss
        term_2_coef = 1 - (dt / t)
        term_2_log_numerator = term_1_log_numerator
        term_2_log_denom = torch.log((alpha_s * x_theta_at_m) / (t - dt) + 1)
        
        L_vb_masked = (term_1_coef * (term_1_log_numerator - term_1_log_denom) + 
                       term_2_coef * (term_2_log_numerator - term_2_log_denom))
        
        # multiply by <zt, m> term
        L_vb = L_vb_masked * (xt == self.mask_token_id)
        
        # scale by T and return
        return self.T * L_vb
    
    def _forward_pass_diffusion(self, x0, attn_mask, bond_mask=None, mask=None):
        """
            Training reverse diffusion model x_theta to reconstruct samples x0
            
            bond_mask: (batch, seq_length)
        """
        # randomly sample time steps to start the denoising process for each x0 in batch
        t = self.sample_t(x0.shape[0], self.device)
        
        # if we are training the intermediate transition blocks
        if self.T > 0: 
            # scale by total timesteps T and cast to integer
            t = (t * self.T).to(torch.int)
            # scale down by T to get a multiple of 1/T
            t = t / self.T
            # add 1/T to ensure no 0 values
            t += (1 / self.T)
        
        # get noise and rate of noise at timestep t
        # sigma = -log(1-t); dsigma = 1 / (1-t)
        sigma, dsigma = self.noise(t)
        time_conditioning = sigma[:, None]
        
        # Get masking probabilities for all tokens for each batch
        # log-linear: 1 - alpha = t
        base_mask_prob = 1 - torch.exp(-sigma[:, None])  # (batch_size, L)

        if self.config.noise.state_dependent and (bond_mask is not None):
            # log-polynomial masking schedule: alpha = 1 - t^w
            # bond_sigma = -log(1-t^w) for w = 3 (default)
            # bond_dsigma = -wt^(w-1) / (1-t^w)
            bond_sigma, bond_dsigma = self.bond_noise(t) # scalar
            # expand dimensions for broadcasting to (B, L)
            bond_sigma = bond_sigma[:, None] 
            bond_dsigma = bond_dsigma[:, None]
            sigma = sigma[:, None]
            dsigma = dsigma[:, None]
            
            # compute masking probability for peptide bonds 1 - bond_alpha = t^w
            bond_mask_prob = 1 - torch.exp(-bond_sigma).to(self.device)
            # piece together (B, L) tensor with modified masking prob at peptide-bond locations
            mask_prob = torch.where(bond_mask == 1, bond_mask_prob, base_mask_prob).to(self.device)
            #print(mask_prob)
            dsigma = torch.where(bond_mask == 1, bond_dsigma, dsigma).to(self.device)
            sigma = torch.where(bond_mask == 1, bond_sigma, sigma).to(self.device)
        else:
            mask_prob = base_mask_prob.to(self.device)
        
        # get masked samples at different timesteps
        if mask is None: 
            zt = self.q_xt(x0, mask_prob).to(self.device)
        else: 
            zt = x0.where(mask==1, torch.full_like(x0, self.mask_token_id)).to(self.device)
        
        model_output = self.forward(zt, attn_mask=attn_mask.to(self.device), sigma=time_conditioning).to(self.device)
        
        # debugging
        assert not torch.isnan(model_output).any()
        assert model_output.is_cuda
        utils.print_nans(model_output, 'model_output')
        
        # compute invalid loss
        invalid_loss = self.compute_invalid_loss(logits=model_output).to(self.device) # (B, L)
        #print(invalid_loss)
        
        if self.T > 0:
            # compute diffusion loss
            diffusion_loss = self.compute_diffusion_loss(model_output, zt, x0, t)
            return diffusion_loss
        
        # compute loss for the final that converts from z0 to x0
        # -log(p_theta)
        # get (batch_size, L) array of log-probabilities 
        log_p_theta = torch.gather(input=model_output, dim=-1, index=x0[:, :, None]).squeeze(-1).to(self.device) # (B, L)
        
        if self.config.noise.state_dependent and (bond_mask is not None):
            return (-log_p_theta * (dsigma / torch.expm1(sigma)) + invalid_loss).to(self.device)
        else: 
            return ((-log_p_theta * (dsigma / torch.expm1(sigma))[:, None]) + invalid_loss).to(self.device)

    def _loss(self, x0, attn_mask, bond_mask=None, mask=None):        
        loss = self._forward_pass_diffusion(x0, attn_mask, bond_mask, mask)
        
        # negative log loss
        nlls = loss * attn_mask
            
        # count number of tokens
        num_tokens = attn_mask.sum()
        
        # compute batch loss
        batch_nll = nlls.sum()
        # compute per token loss
        token_nll = batch_nll / num_tokens
        # return losses
        return Loss(loss = token_nll.to(self.device), nlls = nlls.to(self.device), attn_mask = attn_mask.to(self.device))
    
    def _compute_loss(self, batch, prefix, bond_mask=None):
        
        attn_mask = batch['attention_mask'].to(self.device)
            
        if 'mask' in batch: 
            mask = batch['mask'].to(self.device)
        else: 
            mask = None
        
        if 'bond_mask' in batch:
            bond_mask = batch['bond_mask'].to(self.device)
        else:
            bond_mask = None
        
        losses = self._loss(batch['input_ids'].to(self.device), attn_mask, bond_mask, mask)
        loss = losses.loss

        if prefix == 'train':
            self.train_metrics.update(
                losses.nlls.to(self.device), 
                losses.attn_mask.to(self.device)
            )
            metrics = self.train_metrics
        elif prefix == 'val':
            self.valid_metrics.update(
                losses.nlls.to(self.device), 
                losses.attn_mask.to(self.device)
            )
            metrics = self.valid_metrics
        elif prefix == 'test':
            self.test_metrics.update(losses.nlls, losses.attn_mask)
            metrics = self.test_metrics
        else:
            raise ValueError(f'Invalid prefix: {prefix}')
        
        self.log_dict(metrics,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True)
        
        return loss
        
    
    """SAMPLING"""
    
    def generate_from_masked(self, num_samples=None, seq_length=None, sample_steps=128, eps=1e-5):
        # get number of timesteps
        if sample_steps is None:
            sample_steps = self.config.sampling.steps
        
        if seq_length is None:
            seq_length = self.config.sampling.seq_length

        # sample fully masked sequences
        z = self.sample_prior(num_samples, seq_length).to(self.device)
        
        # create vector of sample_steps timesteps
        timesteps = torch.linspace(1, eps, sample_steps + 1, device=self.device)
        
        # compute interval between timesteps
        dt = (1 - eps) / sample_steps
        
        for i in range(sample_steps):
            t = timesteps[i] * torch.ones(z.shape[0], 1, device=self.device)
            
            z = self.single_reverse_step(z, t, dt)
        
        return z
    
    
    """SAMPLING STEP"""
    
    def single_reverse_step(self, zt, t, dt, attn_mask=None):
        """
            Take a single reverse diffusion step for the expansion step of the MCTS algorithm
        """
        # get sigma values that determine masking prob
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        
        # reshape sigmas
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        
        # compute masking probabilities for each timestep
        change_prob_t = 1 - torch.exp(-sigma_t)
        change_prob_s = 1 - torch.exp(-sigma_s)
        
        # expand dimensions
        change_prob_t = change_prob_t[:, None, None]
        change_prob_s = change_prob_s[:, None, None]
        
        # get prodiction model that outputs token probabilities
        log_p_x0 = self.forward(zt, attn_mask=attn_mask, sigma=sigma_t) 
        
        # check dimensions match
        assert change_prob_t.ndim == log_p_x0.ndim
        
        # compute reverse diffusion probability of being unmasked at timestep s
        # (sigma_s - sigma_t)*x_theta
        q_zs = log_p_x0.exp() * (change_prob_t - change_prob_s)
        
        # compute reverse diffusion probability of remaining masked at timestep s
        # (1 - sigma_s)*m
        q_zs[:, :, self.mask_token_id] = change_prob_s[:, :, 0]
        
        # sample sequence at timestep s from categorical distribution of q_zs
        z_changed = sample_categorical(q_zs)
        
        copy_flag = (zt != self.mask_token_id).to(zt.dtype)
        return (copy_flag * zt) + ((1 - copy_flag) * z_changed)

    def cached_reverse_step(self, x, t, dt, p_x0=None, attn_mask=None):
        assert self.config.noise.type == 'loglinear'
        sigma_t, _ = self.noise(t)
        
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        
        change_prob_t = t[:, None, None]
        change_prob_s = (t - dt)[:, None, None]
        
        assert change_prob_t.ndim == 3, change_prob_t.shape
        
        if p_x0 is None:
            p_x0 = self.forward(x, attn_mask=attn_mask, sigma=sigma_t).exp()
        
        assert change_prob_t.ndim == p_x0.ndim
        
        q_xs = p_x0 * (change_prob_t - change_prob_s)
        
        # zero-masking probability
        q_xs[:, :, self.mask_token_id] = change_prob_s[:, :, 0]
        
        x_changed = sample_categorical(q_xs)
        
        copy_flag = (x != self.mask_token_id).to(x.dtype)
        
        return p_x0, copy_flag * x + (1 - copy_flag) * x_changed
    
    # first step in expansion
    def batch_cached_reverse_step(self, token_array, t, dt, batch_size, p_x0=None, attn_mask=None):
        
        assert self.config.noise.type == 'loglinear'
        sigma_t, _ = self.noise(t)
        
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        
        change_prob_t = t[:, None, None]
        change_prob_s = (t - dt)[:, None, None]
        
        assert change_prob_t.ndim == 3, change_prob_t.shape
        
        if token_array.dim() == 1:
            token_array = token_array.unsqueeze(0)
            #token_array = token_array.repeat(batch_size, 1)
            
        attn_mask = torch.ones_like(token_array)
        
        if p_x0 is None:
            p_x0 = self.forward(token_array, attn_mask=attn_mask, sigma=sigma_t).exp()
        
        assert change_prob_t.ndim == p_x0.ndim
        
        q_xs = p_x0 * (change_prob_t - change_prob_s)
        
        # zero-masking probability
        q_xs[:, :, self.mask_token_id] = change_prob_s[:, :, 0]
        
        # repeat the parent token along the first dimension which will be unmasked into distinct sequences
        token_array = token_array.repeat(batch_size, 1)
        
        if self.config.mcts.sampling == 0:
            x_changed = sample_batched_categorical(q_xs.to(self.device), batch_size)
        else:
            x_changed = sample_batched_top_k(q_xs.to(self.device), batch_size, self.config.mcts.sampling)
        
        copy_flag = (token_array != self.mask_token_id).to(token_array.dtype)
        
        return p_x0, copy_flag * token_array + (1 - copy_flag) * x_changed
    
    def _process_sigma(self, sigma):
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma
    
    def forward(self, zt, attn_mask, sigma):
        """
        Predicts the token log-probabilities from zt at time t with noise schedule sigma
        """
        sigma = self._process_sigma(sigma)
        
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float32, cache_enabled=True):
            logits = self.backbone(zt, attn_mask).to(self.device)
            
        return self.subs_parameterization(logits, zt)
    
    def subs_parameterization(self, logits, zt):
        """
        Updates reverse diffusion logits based on SUBS parameterization:
        - zero masking probabilities: -infinity probability of being masked during reverse diffusion
        - carry-over unmasking: unmasked input tokens remain unchanged during reverse diffusion
        Args:
            logits: vector of token probabilities for unmasking masked tokens
            zt: partially unmasked sequence at current timestep
        """
        logits[:, :, self.mask_token_id] += self.neg_infinity # [sequence index, current token, next token]
        
        
        logits = (logits - torch.logsumexp(logits, dim=-1, keepdim=True)).to(self.device)
        
        
        unmasked_indices = (zt != self.mask_token_id).to(self.device)  # shape: [200, seq_length]
        batch_idx, seq_idx = torch.where(unmasked_indices)  # Get explicit indices
        batch_idx = batch_idx.to(self.device)
        seq_idx = seq_idx.to(self.device)
        tokens = zt[batch_idx, seq_idx].to(self.device)  # Get the tokens at those positions
        
        assert logits.is_contiguous(), "logits tensor is not contiguous"
        assert unmasked_indices.shape == zt.shape, "same shape"
        assert not torch.isnan(logits).any(), "NaN values found in logits"
        assert tokens.max() < logits.shape[-1], "token indices out of bounds"
        assert batch_idx.max() < logits.shape[0], "batch index out of bounds"
        assert seq_idx.max() < logits.shape[1], "seq index out of bounds"
        assert batch_idx.device == seq_idx.device == logits.device == tokens.device, "device inconsistent"

        logits[batch_idx, seq_idx] = self.neg_infinity  # Set everything to -inf first
        logits[batch_idx, seq_idx, tokens] = 0  # Set only the specific token positions to 0
        # return logits with SUBS parameterization
        return logits.to(self.device)
    
    """SAMPLING"""
    @torch.no_grad()
    def _sample(self, num_steps=None, eps=1e-5, x_input=None):
        """ 
            Generate samples
        """
        batch_size_per_gpu = self.config.eval.perplexity_batch_size
        
        if num_steps is None:
            num_steps = self.config.sampling.steps
        
        if x_input is not None:
            x = x_input['input_ids'].to(self.device)
            attn_mask = x_input['attention_mask'].to(self.device)
        else:
            x = self.sample_prior(batch_size_per_gpu, self.config.model.length).to(self.device)
            attn_mask = torch.ones_like(x).to(self.device)
        
        
        timesteps = torch.linspace(1, eps, num_steps+1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None
        generation_history = [] # used to track which tokens are unmasked
        
        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device = self.device)
            if self.sampler == 'ddpm':
                x = self.single_reverse_step(x, t, dt).to(self.device)
                
            elif self.sampler == 'ddpm_cache':
                p_x0_cache, x_next = self.cached_reverse_step(x, t, dt, p_x0=p_x0_cache, attn_mask=attn_mask)
                if (not torch.allclose(x_next, x) or self.time_conditioning):
                    # Disable caching
                    p_x0_cache = None
                x = x_next.to(self.device)
                #print(self.tokenizer.decode(x.squeeze()))
            else:
                x = self._analytic_update(x, t, dt, attn_mask).to(self.device)
        
        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == 'analytic':
                x = self._denoiser_update(x, t).to(self.device)
            else:
                time_conditioning = self.noise(t)[0].to(self.device)
                x = self.forward(x, attn_mask=attn_mask, sigma=time_conditioning).argmax(dim=-1).to(self.device)
                #print(self.tokenizer.decode(x.squeeze()))
        return x.to(self.device)


    def restore_model_and_sample(self, num_steps, eps=1e-5):
        """Generate samples from the model."""
        self.backbone.eval()
        self.noise.eval()
        samples = self._sample(num_steps=num_steps, eps=eps)
        self.backbone.train()
        self.noise.train()
        return samples

    def get_score(self, zt, sigma, attn_mask=None):
        
        # score(x, t) = p_t(y) / p_t(x)
        # => log score(x, t) = log p_t(y) - log p_t(x)
        
        # case 1: x = masked
        #   (i) y = unmasked
        #     log score(x, t) = log p_\theta(x)|_y + log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))
        #   (ii) y = masked
        #     log score(x, t) = 0

        # case 2: x = unmasked
        #   (i) y != masked, y != x
        #     log score(x_i, t) = - inf
        #   (ii) y = x 
        #     log score(x_i, t) = 0
        #   (iii) y = masked token
        #     log score(x_i, t) = - log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))
        
        model_output = self.forward(zt, attn_mask=attn_mask, sigma=sigma)
    
        log_k = -torch.log(torch.expm1(sigma)).squeeze(-1)
        assert log_k.ndim == 1
        
        masked_score = model_output + log_k[:, None, None]
        masked_score[:, :, self.mask_token_id] = 0

        unmasked_score = self.neg_infinity * torch.ones_like(model_output)
        unmasked_score = torch.scatter(
            unmasked_score, -1,
            zt[..., None],
            torch.zeros_like(unmasked_score[..., :1]))
        
        unmasked_score[:, :, self.mask_token_id] = - (log_k[:, None] * torch.ones_like(zt))
        
        masked_indices = (zt == self.mask_token_id).to(model_output.dtype)[:, :, None]
        
        model_output = (masked_score * masked_indices + unmasked_score * (1 - masked_indices))
        
        return model_output.exp()

    def _staggered_score(self, score, dsigma):
        score = score.clone()
        extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., self.mask_token_id] += extra_const
        return score

    def _analytic_update(self, x, t, step_size, attn_mask=None):
        curr_sigma, _ = self.noise(t)
        next_sigma, _ = self.noise(t - step_size)
        dsigma = curr_sigma - next_sigma
        score = self.get_score(x, attn_mask, curr_sigma)
        stag_score = self._staggered_score(score, dsigma)
        probs = stag_score * self._transp_transition(x, dsigma)
        return sample_categorical(probs)

    def _denoiser_update(self, x, t):
        sigma, _ = self.noise(t)
        score = self.get_score(x, sigma)
        stag_score = self._staggered_score(score, sigma)
        probs = stag_score * self._transp_transition(x, sigma)
        probs[..., self.mask_token_id] = 0
        samples = sample_categorical(probs)
        return samples

    def _transp_transition(self, i, sigma):
        sigma = unsqueeze(sigma, reference=i[..., None])
        edge = torch.exp(-sigma) * F.one_hot(
        i, num_classes=self.vocab_size)
        edge += torch.where(i == self.mask_token_id,
                            1 - torch.exp(-sigma).squeeze(-1),
                            0)[..., None]
        return edge   
        
        
    def on_train_epoch_start(self):
        torch.cuda.empty_cache()
        self.backbone.train()
        self.noise.train()
    
    
    def training_step(self, batch, batch_idx):
        # Initialize throughput calculation
        start_time = time.time()

        if self.config.vocab == 'old_smiles' or self.config.vocab == 'new_smiles':
            loss = self._compute_loss(batch, prefix='train', bond_mask=batch['bond_mask'])
        else:
            loss = self._compute_loss(batch, prefix='train')
            
        self.log(name='trainer/loss',
                value=loss.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=True)
        
        # Calculate throughput
        elapsed_time = time.time() - start_time
        total_tokens = batch['input_ids'].numel()
        throughput = total_tokens / elapsed_time

        self.log(name='trainer/throughput',
                value=throughput,
                on_step=True,
                on_epoch=False,
                sync_dist=True)

        return loss
    

    def on_load_checkpoint(self, checkpoint):
        self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
        self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']
    
    """VALIDATION"""
    def on_validation_epoch_start(self):
        gc.collect()
        torch.cuda.empty_cache()
        self.backbone.eval()
        self.noise.eval()
        assert self.valid_metrics.nll.mean_value == 0
        assert self.valid_metrics.nll.weight == 0

    def validation_step(self, batch, batch_idx):
        if self.config.vocab == 'old_smiles' or self.config.vocab == 'new_smiles':
            loss = self._compute_loss(batch, prefix='val', bond_mask=batch['bond_mask'])
        else:
            loss = self._compute_loss(batch, prefix='val')
            
        self.log(name='trainer/val_loss',
                value=loss.item(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        gc.collect()
        torch.cuda.empty_cache()

    """OPTIMIZATION"""

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        gc.collect()
        torch.cuda.empty_cache()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            itertools.chain(self.backbone.parameters(),self.noise.parameters()),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay
        )
            
        self.total_steps = self.config.trainer.max_steps
        scheduler = CosineWarmup(optimizer,
                                warmup_steps=self.config.lr_scheduler.num_warmup_steps,
                                total_steps=self.total_steps)

        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'monitor': 'val/loss',
            'name': 'trainer/lr'
        }

        return [optimizer], [scheduler_dict]

    @torch.no_grad()
    def compute_masked_perplexity(self, generated_ids, input_ids):
        """
            Computes masked perplexity between array of generated token ids and masked ids that are converted to logits
        """
        
        total_nll = 0
        total_tokens = 0
        
        input_ids = torch.tensor(input_ids).to(self.device)
        #print(input_ids)

        for sequence in generated_ids:
            # tokenize the sequence
            
            gt_ids = torch.tensor(sequence).to(self.device)
            #print(gt_ids)

            sys.stdout.flush()

            # forward pass thorugh backbone peptideclm model
            attn_mask = torch.ones_like(input_ids).to(self.device)
            
            # compute logits using backbone
            
            if self.config.mode in ['train', 'ppl_eval']:
                outputs = self.backbone.forward(input_ids=input_ids, attn_mask=attn_mask)
            elif self.config.mode == 'sample_eval':
                outputs = self.backbone.forward(input_ids=input_ids)
            
            
            # get logits for each position in sequence across all tokens in vocab
            #logits = outputs[-1] # (batch_size, seq_length, vocab_size)

            logits = outputs.view(-1, outputs.size(-1))  
            gt_ids = gt_ids.view(-1)    
            
            #print(logits.shape)
            #print(gt_ids.shape)

            # compute loss
            # shift_logits = logits[:, :-1, :].contiguous() # remove eos
            # shift_labels = input_ids[:, 1:].contiguous()
            # print(masked)
            
            loss = F.cross_entropy(logits, 
                                    gt_ids.where(input_ids==self.mask_token_id, torch.full_like(gt_ids, -100)).view(-1), 
                                    reduction='sum')

            total_nll += loss.item()
            # count all non-padding tokens
            total_tokens += input_ids.ne(self.tokenizer.pad_token_id).sum().item() # count in bos and eos
            
        # compute pseudo-perplexity
        # print(total_nll, ",;,", total_tokens)
        pseudo_perplexity = torch.exp(torch.tensor(total_nll / total_tokens))
        self.gen_ppl_metric.update(pseudo_perplexity)

        return pseudo_perplexity.item()