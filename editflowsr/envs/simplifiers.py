import traceback
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from .generators import all_operators
import numpy as np
from functools import partial
import numexpr as ne
import sympytorch
import torch


class InvalidPrefixExpression(BaseException):
    pass


class Simplifier:
    def __init__(self, generator):

        self.params = generator.params
        self.encoder = generator.equation_encoder
        self.operators = generator.operators
        self.max_int = generator.max_int
        self.local_dict = {
            "n": sp.Symbol("n", real=True, nonzero=True, positive=True, integer=True),
            "e": sp.E,
            "pi": sp.pi,
            "euler_gamma": sp.EulerGamma,
            "arcsin": sp.asin,
            "arccos": sp.acos,
            "arctan": sp.atan,
            "step": sp.Heaviside,
            "sign": sp.sign,
        }
        for k in generator.variables:
            self.local_dict[k] = sp.Symbol(k, real=True, integer=False)

    def tree_to_sympy_expr(self, tree):
        prefix = tree.prefix().split(",")
        sympy_compatible_infix = self.prefix_to_sympy_compatible_infix(prefix)
        expr = parse_expr(
            sympy_compatible_infix, evaluate=True, local_dict=self.local_dict
        )
        return expr

    def tree_to_torch_module(self, tree, dtype=torch.float32):
        expr = self.tree_to_sympy_expr(tree)
        mod = self.expr_to_torch_module(expr, dtype)
        return mod

    def expr_to_torch_module(self, expr, dtype):
        mod = sympytorch.SymPyModule(expressions=[expr])
        mod.to(dtype)

        def wrapper_fn(_mod, x, constants=None):
            local_dict = {}
            x_var_names = [str(s) for s in expr.free_symbols if str(s).startswith("x_")]
            for name in x_var_names:
                    d = int(name[2:])
                    if d < x.shape[1]:
                        local_dict[name] = x[:, d]
                    else:
                        local_dict[name] = torch.zeros(x.shape[0], dtype=x.dtype)
            if constants is not None:
                for d in range(constants.shape[0]):
                    local_dict["CONSTANT_{}".format(d)] = constants[d]
            return _mod(**local_dict)

        return partial(wrapper_fn, mod)

    def tree_to_numexpr_fn(self, tree):
        infix = tree.infix()
        numexpr_equivalence = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "pow": "**",
            "inv": "1/",
        }

        for old, new in numexpr_equivalence.items():
            infix = infix.replace(old, new)

        def get_vals(dim, val):
            vals_ar = np.empty((dim,))
            vals_ar[:] = val
            return vals_ar

        def wrapped_numexpr_fn(_infix, x, extra_local_dict={}):
            assert isinstance(x, np.ndarray) and len(x.shape) == 2
            local_dict = {}
            for d in range(self.params.max_input_dimension):
                if "x_{}".format(d) in _infix:
                    if d >= x.shape[1]:
                        local_dict["x_{}".format(d)] = np.zeros(x.shape[0])
                    else:
                        local_dict["x_{}".format(d)] = x[:, d]
            local_dict.update(extra_local_dict)
            try:
                vals = ne.evaluate(_infix, local_dict=local_dict)
                if len(vals.shape) == 0:
                    vals = get_vals(x.shape[0], vals)
            except Exception as e:
                print(e)
                print("problem with tree", _infix)
                traceback.format_exc()
                vals = get_vals(x.shape[0], np.nan)
            return vals[:, None]

        return partial(wrapped_numexpr_fn, infix)

    def write_infix(self, token, args):
        if token == "add":
            return f"({args[0]})+({args[1]})"
        elif token == "sub":
            return f"({args[0]})-({args[1]})"
        elif token == "mul":
            return f"({args[0]})*({args[1]})"
        elif token == "div":
            return f"({args[0]})/({args[1]})"
        if token == "pow":
            return f"({args[0]})**({args[1]})"
        elif token == "idiv":
            return f"idiv({args[0]},{args[1]})"
        elif token == "mod":
            return f"({args[0]})%({args[1]})"
        elif token == "abs":
            return f"Abs({args[0]})"
        elif token == "inv":
            return f"1/({args[0]})"
        elif token == "pow2":
            return f"({args[0]})**2"
        elif token == "pow3":
            return f"({args[0]})**3"
        elif token in all_operators:
            return f"{token}({args[0]})"
        else:
            return token

    def _prefix_to_sympy_compatible_infix(self, expr):
        if len(expr) == 0:
            raise InvalidPrefixExpression("Empty prefix list.")
        t = expr[0]
        if t in all_operators:
            args = []
            l1 = expr[1:]
            for _ in range(all_operators[t]):
                i1, l1 = self._prefix_to_sympy_compatible_infix(l1)
                args.append(i1)
            return self.write_infix(t, args), l1
        else:
            try:
                float(t)
                t = str(t)
            except ValueError:
                t = t
            return t, expr[1:]

    def prefix_to_sympy_compatible_infix(self, expr):
        p, r = self._prefix_to_sympy_compatible_infix(expr)
        if len(r) > 0:
            raise InvalidPrefixExpression(
                f'Incorrect prefix expression "{expr}". "{r}" was not parsed.'
            )
        return f"({p})"
