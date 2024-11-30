from typing import List, Union


class Tokenizer:
    def __init__(self, raw_str_data: str) -> None:
        self.vocab = sorted(list(set(raw_str_data)))
        self.vocab_size = len(self.vocab)
        self.str_to_id = {char: i for i, char in enumerate(self.vocab)}
        self.id_to_str = {i: char for i, char in enumerate(self.vocab)}

    def encode(self, text: Union[str, List[str]]) -> Union[List[List[str]], List[str]]:
        if isinstance(text, str):
            return [self.str_to_id[ch] for ch in text]
        else:
            return [self.encode(s) for s in text]

    def decode(
        self, input_ids: Union[List[int], List[List[int]]]
    ) -> Union[str, List[str]]:
        if isinstance(input_ids[0], int):
            return "".join([self.id_to_str[id] for id in input_ids])
        else:
            return [self.decode(ids) for ids in input_ids]
