from logging import getLogger
import numpy as np
from . import generators, simplifiers


SPECIAL_WORDS = [
    "<EOS>",
    "<X>",
    "</X>",
    "<Y>",
    "</Y>",
    "</POINTS>",
    "<INPUT_PAD>",
    "<OUTPUT_PAD>",
    "<PAD>",
    "(",
    ")",
    "SPECIAL",
    "OOD_unary_op",
    "OOD_binary_op",
    "OOD_constant",
]


logger = getLogger()


class FunctionEnvironment:
    def __init__(self, params):
        self.params = params
        self.rng = None
        self.float_precision = params.float_precision
        self.mantissa_len = params.mantissa_len
        self.max_size = None
        assert (
            params.float_precision + 1
        ) % params.mantissa_len == 0, "Bad precision/mantissa len ratio"

        self.generator = generators.RandomFunctions(params, SPECIAL_WORDS)
        self.float_encoder = self.generator.float_encoder
        self.float_words = self.generator.float_words
        self.equation_encoder = self.generator.equation_encoder
        self.equation_words = self.generator.equation_words
        self.equation_words += self.float_words

        self.simplifier = simplifiers.Simplifier(self.generator)


        self.float_id2word = {i: s for i, s in enumerate(self.float_words)}
        self.equation_id2word = {i: s for i, s in enumerate(self.equation_words)}
        self.float_word2id = {s: i for i, s in self.float_id2word.items()}
        self.equation_word2id = {s: i for i, s in self.equation_id2word.items()}

        for ood_unary_op in self.generator.extra_unary_operators:
            self.equation_word2id[ood_unary_op] = self.equation_word2id["OOD_unary_op"]
        for ood_binary_op in self.generator.extra_binary_operators:
            self.equation_word2id[ood_binary_op] = self.equation_word2id[
                "OOD_binary_op"
            ]
        if self.generator.extra_constants is not None:
            for c in self.generator.extra_constants:
                self.equation_word2id[c] = self.equation_word2id["OOD_constant"]

        assert len(self.float_words) == len(set(self.float_words))
        assert len(self.equation_word2id) == len(set(self.equation_word2id))
        self.n_words = params.n_words = len(self.equation_words)
        logger.info(
            f"vocabulary: {len(self.float_word2id)} float words, {len(self.equation_word2id)} equation words"
        )

    def word_to_infix(self, words, is_float=True, str_array=True):
        if is_float:
            m = self.float_encoder.decode(words)
            if m is None:
                return None
            if str_array:
                return np.array2string(np.array(m))
            else:
                return np.array(m)
        else:
            m = self.equation_encoder.decode(words)

            if m is None:
                return None
            if str_array:
                return m.infix()
            else:
                return m

    def wrap_equation_floats(self, tree, constants):
        prefix = tree.prefix().split(",")
        j = 0
        for i, elem in enumerate(prefix):
            if elem.startswith("CONSTANT"):
                prefix[i] = str(constants[j])
                j += 1
        assert j == len(constants), "all constants were not fitted"
        assert "CONSTANT" not in prefix, "tree {} got constant after wrapper {}".format(
            tree, constants
        )
        tree_with_constants = self.word_to_infix(
            prefix, is_float=False, str_array=False
        )
        return tree_with_constants
