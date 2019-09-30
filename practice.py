from utils import getmodel

import argparse

from transformer import Transformer



model=getmodel(model=Transformer,eval_after=300,eval_step=100)