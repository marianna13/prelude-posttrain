"""Prelude's pre-rendered mid-training rows on the existing Levanter chat loader."""
from dataclasses import dataclass

from levanter.data.text.formats import LmDatasetFormatBase
from scripts_grug.grug_datakit_chat import SnowballPrerenderedChatFormat
from scripts_grug.grug_datakit_chat import PrerenderedRowProcessor


@LmDatasetFormatBase.register_subclass("prelude_midtrain")
@dataclass(frozen=True)
class PreludeMidtrainFormat(SnowballPrerenderedChatFormat):
    # Ordinary user turns train in this format. The field name is inherited from
    # Levanter: True means consume the precomputed mask, not recompute by role.
    mask_user_turns: bool = True
    slice_strategy: str = "raise"

    def build_preprocessor(self, tokenizer, *, enforce_eos=True, enforce_bos=True):
        if self.slice_strategy != "raise":
            raise ValueError("Prelude mid-training forbids implicit truncation")
        if len(tokenizer) != 262151 or tokenizer.bos_token_id != 1 or tokenizer.pad_token_id != 262144:
            raise ValueError("Prelude mid-training requires its frozen tokenizer")
        if not self.mask_user_turns:
            raise ValueError('Prelude mid-training requires mask_user_turns=True')
        return PrerenderedRowProcessor(ids_field=self.ids_field, loss_field=self.loss_field,
            max_tokens=self.max_tokens, vocab_size=len(tokenizer), bos_id=tokenizer.bos_token_id,
            allow_zero_loss=True)
