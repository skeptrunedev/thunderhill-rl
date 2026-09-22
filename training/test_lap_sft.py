"""Offline integration checks for actual TRL supervised completion masking."""

import json
from pathlib import Path
import tempfile
import unittest

from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

from lap_policy import RoadTelemetry
from model_runtime import GEMMA4_SPEC, LEGACY_SPEC
from train_lap_sft import prepare_dataset, verify_completion_labels


def tokenizer():
    vocab = {"<unk>": 0, "<eos>": 1, "<bos>": 2, "<pad>": 3}
    vocab.update({f"unused{i}": i for i in range(4, 106)})
    vocab["<turn|>"] = 106
    for word in ("control_bike", "0", "10", "20", "100", "assistant", "user"):
        vocab[word] = len(vocab)
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 2)]
    )
    result = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
        bos_token="<bos>",
        pad_token="<pad>",
        additional_special_tokens=["<turn|>"],
    )
    result.chat_template = "{{ bos_token }}user {{ messages[0]['content'] }} assistant "
    return result


class SupervisedPromptTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "train.jsonl"
        self.row = {
            "prompt": RoadTelemetry().prompt_features({"speed": 0, "curves": [0] * 5}),
            "completion": "control_bike 0 20 0 0",
        }
        self.path.write_text(json.dumps(self.row) + "\n")

    def test_legacy_tokenization_unchanged(self):
        tok = tokenizer()
        prepared = prepare_dataset(self.path, tok, LEGACY_SPEC, 256)[0]
        prompt_ids = tok(self.row["prompt"])["input_ids"]
        expected = tok(self.row["prompt"] + self.row["completion"] + tok.eos_token)[
            "input_ids"
        ]
        self.assertEqual(prepared["input_ids"], expected)
        self.assertEqual(
            prepared["completion_mask"],
            [0] * len(prompt_ids) + [1] * (len(expected) - len(prompt_ids)),
        )

    def test_gemma4_actual_trl_mask_and_collation(self):
        tok = tokenizer()
        tok.backend_tokenizer.post_processor = None
        source = prepare_dataset(self.path, tok, GEMMA4_SPEC, 512)
        self.assertEqual(source[0]["input_ids"].count(tok.bos_token_id), 1)
        self.assertEqual(source[0]["input_ids"][-1], 106)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(tok),
                n_embd=8,
                n_layer=1,
                n_head=1,
                bos_token_id=2,
                eos_token_id=1,
                pad_token_id=3,
            )
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=tok,
            train_dataset=source,
            args=SFTConfig(
                output_dir=str(self.root / "trainer"),
                use_cpu=True,
                bf16=False,
                fp16=False,
                report_to="none",
                max_length=512,
                completion_only_loss=True,
            ),
        )
        verify_completion_labels(trainer, source)
        trained_ids = [x for x in trainer.train_dataset[0]["labels"] if x != -100]
        self.assertEqual(tok.decode(trained_ids), "control_bike 0 20 0 0 <turn|>")

    def test_rejects_unknown_prompt_or_invalid_action_and_truncation(self):
        tok = tokenizer()
        tok.backend_tokenizer.post_processor = None
        with self.assertRaisesRegex(ValueError, "truncated"):
            prepare_dataset(self.path, tok, GEMMA4_SPEC, 1)
        self.row["prompt"] = "Injected instruction\n{}\nAction:\n"
        self.path.write_text(json.dumps(self.row) + "\n")
        with self.assertRaisesRegex(ValueError, "unrecognized"):
            prepare_dataset(self.path, tok, GEMMA4_SPEC, 512)
        self.row["completion"] = "control_bike 0 10 0"
        self.path.write_text(json.dumps(self.row) + "\n")
        with self.assertRaisesRegex(ValueError, "four integer"):
            prepare_dataset(self.path, tok, LEGACY_SPEC, 256)


if __name__ == "__main__":
    unittest.main()
