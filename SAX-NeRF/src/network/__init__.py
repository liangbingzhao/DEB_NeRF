from .network import DensityNetwork
from .Lineformer import Lineformer
from .Lineformer_dual import Lineformer_dual
from .Lineformer_singlefield import Lineformer_singlefield
from .Lineformer_basis2 import Lineformer_basis2


def get_network(type):
    if type == "mlp":
        return DensityNetwork
    elif type == "Lineformer":
        return Lineformer
    elif type == "Lineformer_dual":
        return Lineformer_dual
    elif type == "Lineformer_singlefield":
        return Lineformer_singlefield
    elif type == "Lineformer_basis2":
        return Lineformer_basis2
    else:
        raise NotImplementedError("Unknown network type!")

