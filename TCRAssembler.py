class Assembler():
    """Assembels TCR sequence and structure from cdr3 sequences"""

    def __init__(self, cdr, v_gene='', d_gene='', j_gene=''):
        """Constructor for Assembler"""
        self._cdr = cdr
        self._v = v_gene
        self._d = d_gene
        self._j = j_gene

