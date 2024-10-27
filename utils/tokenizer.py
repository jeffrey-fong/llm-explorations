from typing import List, Union

import tiktoken


class Tokenizer:
    def __init__(self) -> None:
        self.enc = tiktoken.get_encoding("o200k_base")  # Use o1/gpt-4o encoding
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: Union[str, List[str]]) -> Union[List[List[str]], List[str]]:
        if isinstance(text, str):
            return self.enc.encode(text)
        else:
            return self.enc.encode_batch(text)

    def decode(
        self, input_ids: Union[List[int], List[List[int]]]
    ) -> Union[str, List[str]]:
        if isinstance(input_ids[0], int):
            return self.enc.decode(input_ids)
        else:
            return self.enc.decode_batch(input_ids)
