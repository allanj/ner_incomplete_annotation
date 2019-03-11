class Span:

    def __init__(self, left: int, right: int, type: str, inst_id: int = None):
        self.left = left
        self.right = right
        self.type = type
        self.inst_id = inst_id

    def __eq__(self, other):
        curr = self.left == other.left and self.right == other.right and self.type == other.type
        if self.inst_id is None:
            return curr
        else:
            return curr and self.inst_id == other.inst_id

    def __hash__(self):
        if self.inst_id is None:
            return hash((self.left, self.right, self.type))
        else:
            return hash((self.inst_id, self.left, self.right, self.type))

