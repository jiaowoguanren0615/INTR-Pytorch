from .backbone import *
from .intr import *
from .transformer import *
from .position_encoding import *

def build_model(args):
    return build(args)