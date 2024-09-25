# Object-oriented interface for working w/ file and directory paths
from pathlib import Path

# SentencePieceProcessor is used for tokenizing text
# SentencePieceTrainer is used for training SentencePiece models
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer

# List of available model types for SentencePiece tokenizer
__model_types = ["unigram", "bpe", "word", "char"]


class Tokenizer:
    # Constructor for Tokenizer class
    def __init__(self, path: str = None) -> None:
        # Initialize instance of SentenecePieceProcessor
        self.sp = SentencePieceProcessor()
        # Load a model from the provided path
        self.sp.Load(model_file=path)

    # Define property 'vocab_size' that returns vocab size of loaded model
    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()

    # Define property 'bos_id' that returns ID for beginning-of-sequence token
    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    # Define property 'eos_id' that returns ID for end-of-sequence token
    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    # Define property 'pad_id' that returns ID for padding token
    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    # Define property 'unk_id' that returns ID for unknown token
    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    # Method that takes a string and returns a list of token IDs
    def encode(
        self,
        # Take parameter 'input' of type str
        input: str,
        # Optional bool params indicating whether to add beginning and end-of-string tokens
        beg_of_string: bool = False,
        end_of_string: bool = False,
        # Optional bool params indicating whether to pad the sequence to the desired length
        pad_seq: bool = False,
        # Optional int param specifying desired sequence length for padding
        seq_len: int = None,
        # Indicates method returns list of ints
    ) -> list[int]:
        # Encode the input string as a list of token IDs
        out = self.sp.EncodeAsIds(input, add_bos=beg_of_string, add_eos=end_of_string)
        # If padding is enabled, pad the sequence to the desired length
        if pad_seq and len(out) < seq_len:
            out = [*[self.pad_id] * (seq_len - len(out)), *out]
        # Return the encoded sequence
        return out

    # Method that takes a list of token IDs and returns corresponding decoded string
    def decode(self, input: list[int]) -> str:
        return "".join(self.sp.Decode(input))

# Method that trains SentencePiece model w/ given parameters
def train_tokenizer(
    # Specify path to input file containing text data for training
    input_file: str,
    # Specify desired vocab size for the tokenizer
    vocab_size: int,
    # Optional int params specifying IDs for padding, unknown, beginning-of-string, and end-of-string tokens
    pad_id: int = 0,
    unk_id: int = 1,
    bod_id: int = 2,
    eos_id: int = 3,
    # Optional str param specifying type of SentencePiece model to train
    model_type: str = "unigram",
    # Optional int param specifying max # of lines to read from input file
    max_sample_size: int = 1_000_000,
    # Indicates method doesn't return anything
) -> None:
    # Check that model_type is valid
    assert model_type in __model_types, f"Got invalid model_type argument: {model_type}"
    # Initiate training process for SentencePiece model
    SentencePieceTrainer.Train(
        # Specify path to input file containing text data for training
        input=input_file,
        # Specify desired vocab size for tokenizer
        vocab_size=vocab_size,
        # Specify type of SentencePiece model to train
        model_type=model_type,
        # Specify prefix for model files based on input file path
        model_prefix=Path(input_file).with_suffix(""),
        # Specify IDs for padding, unknown, beginning-of-string, and end-of-string tokens
        pad_id=pad_id,
        unk_id=unk_id,
        bos_id=bod_id,
        eos_id=eos_id,
        # Specify max # of lines to read from input file
        input_sentence_size=max_sample_size,
    )
