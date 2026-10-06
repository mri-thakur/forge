"""Small reverse-mode engine for the CPU training/reference path.

This is deliberately not a general-purpose framework. Operations required by
Forge's transformer have explicit vector-Jacobian products. Float64 finite
differences in tests check them independently of training loss.
"""

from __future__ import annotations

import numpy as np


def unbroadcast(gradient, shape):
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient.reshape(shape)


class Tensor:
    def __init__(self, data, requires_grad=False, parents=(), backward=None):
        self.data = np.asarray(data)
        self.requires_grad = requires_grad
        self.grad = None
        self.parents = parents
        self._backward = backward

    def accumulate(self, gradient):
        if self.requires_grad:
            self.grad = gradient if self.grad is None else self.grad + gradient

    def backward(self):
        if self.data.size != 1:
            raise ValueError("backward requires a scalar")
        order, seen = [], set()

        def visit(node):
            if id(node) not in seen:
                seen.add(id(node))
                for parent in node.parents:
                    visit(parent)
                order.append(node)

        visit(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(order):
            if node._backward and node.grad is not None:
                node._backward(node.grad)

    @staticmethod
    def wrap(value):
        return value if isinstance(value, Tensor) else Tensor(value)

    def __add__(self, other):
        other = self.wrap(other)

        def backward(g):
            self.accumulate(unbroadcast(g, self.data.shape))
            other.accumulate(unbroadcast(g, other.data.shape))

        return Tensor(
            self.data + other.data,
            self.requires_grad or other.requires_grad,
            (self, other),
            backward,
        )

    __radd__ = __add__

    def __mul__(self, other):
        other = self.wrap(other)

        def backward(g):
            self.accumulate(unbroadcast(g * other.data, self.data.shape))
            other.accumulate(unbroadcast(g * self.data, other.data.shape))

        return Tensor(
            self.data * other.data,
            self.requires_grad or other.requires_grad,
            (self, other),
            backward,
        )

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -self.wrap(other)

    def __truediv__(self, other):
        return self * self.wrap(other).power(-1)

    def power(self, exponent):
        def backward(g):
            self.accumulate(g * exponent * self.data ** (exponent - 1))

        return Tensor(self.data**exponent, self.requires_grad, (self,), backward)

    def __matmul__(self, other):
        other = self.wrap(other)

        def backward(g):
            self.accumulate(unbroadcast(g @ np.swapaxes(other.data, -1, -2), self.data.shape))
            other.accumulate(unbroadcast(np.swapaxes(self.data, -1, -2) @ g, other.data.shape))

        return Tensor(
            self.data @ other.data,
            self.requires_grad or other.requires_grad,
            (self, other),
            backward,
        )

    def reshape(self, *shape):
        return Tensor(
            self.data.reshape(*shape),
            self.requires_grad,
            (self,),
            lambda g: self.accumulate(g.reshape(self.data.shape)),
        )

    def transpose(self, *axes):
        inverse = np.argsort(axes)
        return Tensor(
            self.data.transpose(axes),
            self.requires_grad,
            (self,),
            lambda g: self.accumulate(g.transpose(inverse)),
        )

    def __getitem__(self, key):
        def backward(g):
            result = np.zeros_like(self.data)
            np.add.at(result, key, g)
            self.accumulate(result)

        return Tensor(self.data[key], self.requires_grad, (self,), backward)

    def sum(self, axis=None, keepdims=False):
        def backward(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.accumulate(np.broadcast_to(g, self.data.shape))

        return Tensor(
            self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad, (self,), backward
        )

    def mean(self, axis=None, keepdims=False):
        size = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis, keepdims) * (1 / size)

    def sigmoid(self):
        # Clipping avoids overflow in the negative tail; derivative uses the output.
        result = 1 / (1 + np.exp(-np.clip(self.data, -80, 80)))
        return Tensor(
            result,
            self.requires_grad,
            (self,),
            lambda g: self.accumulate(g * result * (1 - result)),
        )


def softmax(x, mask=None):
    data = x.data if mask is None else np.where(mask, x.data, -1e30)
    values = np.exp(data - data.max(axis=-1, keepdims=True))
    result = values / values.sum(axis=-1, keepdims=True)

    def backward(g):
        gradient = result * (g - (g * result).sum(axis=-1, keepdims=True))
        x.accumulate(gradient if mask is None else np.where(mask, gradient, 0))

    return Tensor(result, x.requires_grad, (x,), backward)


def cross_entropy(logits, targets):
    flat = logits.data.reshape(-1, logits.data.shape[-1])
    targets = np.asarray(targets).reshape(-1)
    shifted = flat - flat.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    normalizer = probs.sum(axis=-1, keepdims=True)
    loss = (np.log(normalizer[:, 0]) - shifted[np.arange(len(targets)), targets]).mean()

    def backward(g):
        gradient = probs / normalizer
        gradient[np.arange(len(targets)), targets] -= 1
        logits.accumulate((gradient / len(targets)).reshape(logits.data.shape) * g)

    return Tensor(loss, logits.requires_grad, (logits,), backward)
