import torch

from .bounding_box import  BoxList

class RelationTriplet:
    def __init__(self, instance:BoxList,
                 pair_mat:torch.Tensor, phrase_label:torch.Tensor, phrase_score:torch.Tensor):
        """

        :param sub: boxlist
        :param obj: boxlist
        :param pair_mat: shape (connection_num, 2) [sub, obj]
        :param phrase_label: phrase label_id
        :param phrase_score: phrase label_id
        """

        self.instance = instance

        assert len(pair_mat) == len(phrase_label)
        assert len(phrase_label) == len(phrase_score)
        self.pair_mat = pair_mat
        self.phrase_l = phrase_label
        self.phrase_s = phrase_score

        self.extra_fields = {}

    def to(self, device):
        triplet = RelationTriplet(self.instance.to(device),
                                   self.pair_mat.to(device),
                                   self.phrase_l.to(device),
                                   self.phrase_s.to(device))

        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            triplet.add_field(k, v)
        return triplet


    def get_instance_list(self, side):
        assert side in ['sub', 'obj']
        if side == 'sub':
            return self.instance[self.pair_mat[: ,0]]
        else:
            return self.instance[self.pair_mat[:, 1]]



    """
    add extra information to Box
    """
    def add_field(self, field, field_data):
        assert len(field_data) == len(self.pair_mat)
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())


    def __getitem__(self, item):
        triplet = RelationTriplet(self.instance,
                                   self.pair_mat[item],
                                   self.phrase_l[item],
                                   self.phrase_s[item])
        for k, v in self.extra_fields.items():
            triplet.add_field(k, v[item])
        return triplet


    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_relation={}, ".format(len(self.pair_mat))
        s += "instance_num={}, ".format(len(self.instance))
        s += ")"
        return s