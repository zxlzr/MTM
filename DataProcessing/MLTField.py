from torchtext.data import Field
from torchtext.data import Dataset
import torch
import torch.utils.data
from torch.autograd import Variable
from torchtext.vocab import Vocab

from collections import Counter
from collections import OrderedDict

class MTLField(Field):
    """Defines a datatype together with instructions for converting to Tensor.
    Every dataset consists of one or more types of data. For instance, a text
    classification dataset contains sentences and their classes, while a
    machine translation dataset contains paired examples of text in two
    languages. Each of these types of data is represented by a Field object,
    which holds a Vocab object that defines the set of possible values for
    elements of the field and their corresponding numerical representations.
    The Field object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method and the kind of
    Tensor that should be produced.
    If a Field is shared between two columns in a dataset (e.g., question and
    answer in a QA dataset), then they will have a shared vocabulary.
    Attributes:
        sequential: Whether the datatype represents sequential data. Default:
            True.
        use_vocab: Whether to use a Vocab object. If False, the data in this
            field should already be numerical. Default: True.
        init_token: A token that will be prepended to every example using this
            field, or None for no initial token. Default: None.
        eos_token: A token that will be appended to every example using this
            field, or None for no end-of-sentence token. Default: None.
        fix_length: A fixed length that all examples using this field will be
            padded to, or None for flexible sequence lengths. Default: None.
        tensor_type: The torch.Tensor class that represents a batch of examples
            of this kind of data. Default: torch.LongTensor.
        preprocessing: The Pipeline that will be applied to examples
            using this field after tokenizing but before numericalizing. Many
            Datasets replace this attribute with a custom preprocessor.
            Default: the identity pipeline.
        postprocessing: A Pipeline that will be applied to examples using
            this field after numericalizing but before the numbers are turned
            into a Tensor. Default: the identity pipeline.
        lower: Whether to lowercase the text in this field. Default: False.
        tokenize: The function used to tokenize strings using this field into
            sequential examples. Default: str.split.
    """

    def __init__(
            self, **kwargs):
        super(MTLField, self).__init__(**kwargs)

    def build_vocab(self, dataset_list, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.
        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        counter = Counter()
        sources = []
        for arg in dataset_list:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                counter.update(x)
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.pad_token, self.init_token, self.eos_token]
            if tok is not None))
        # print(counter)
        self.vocab = Vocab(counter, specials=specials, **kwargs)