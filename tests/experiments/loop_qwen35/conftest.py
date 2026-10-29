# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Local tokenizer fixture; tests never download model assets."""

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
from torchtitan.experiments.loop_qwen35.dpo.encoding import Qwen35PreferenceEncoder


@pytest.fixture
def template(tmp_path):
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.decoder = decoders.ByteLevel()
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        ["Hello, history. good bad answer question"],
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=["<unk>", "<pad>", "<|im_start|>", "<|im_end|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<|im_end|>",
    )
    # A fixture-owned model prompt exercises external instruction loading.
    tokenizer.chat_template = (
        "{% if tools %}<|im_start|>system\nFixture tool instructions\n<tools>\n"
        "{{ tools | tojson }}\n</tools>"
        "{% if messages[0].role == 'system' %}\n\n{{ messages[0].content }}{% endif %}"
        "<|im_end|>\n{% endif %}"
        "{% for message in messages if message.role == 'user' %}"
        "<|im_start|>user\n{{ message.content }}<|im_end|>\n{% endfor %}"
    )
    tokenizer.save_pretrained(tmp_path)
    return Qwen35PreferenceEncoder(str(tmp_path), "last_round")
