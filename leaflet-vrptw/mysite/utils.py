class IdMap:
    """maping dari string ke id dan sebaliknya """
    def __init__(self):
        self.str_to_id = {}
        self.id_to_str = []

    def __len__(self):
        return len(self.id_to_str)

    def _get_str(self, i):
        return self.id_to_str[i]

    def _get_id(self, s):
        if s not in self.str_to_id:
            self.str_to_id[s] = len(self.id_to_str)
            self.id_to_str.append(s)
        return self.str_to_id.get(s)
    def __delitem__(self, key):
        if type(key) is str:
            del self.str_to_id[key]

    def __getitem__(self, key):
        if type(key) is int:
            return self._get_str(key)
        elif type(key) is str:
            return self._get_id(key)
        else:
            raise TypeError

    def __setitem__(self, key, val):
        if type(key) is int:
            self.id_to_str[key] = val
        else:
            raise TypeError