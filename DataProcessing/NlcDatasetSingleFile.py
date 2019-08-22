import os

import torchtext.data as data
from torchtext.data import Dataset

class NlcDatasetSingleFile(Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, train, path, text_field, label_field,
                 fine_grained=False, **kwargs):
        """Create an NLC dataset instance given a path and fields.
        Arguments:
            path: Path to the data file.
            text_field: The field that will be used for text data.
            newline_eos: Whether to add an <eos> token for every newline in the
                data file. Default: True.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ('label', label_field)]

        # def get_label_str(label):
        #     return str(int(label) - 1)
        # label_field.preprocessing = data.Pipeline(get_label_str)

        with open(os.path.expanduser(path)) as f_data:
            examples = []
            for line in f_data:
                pair = line.strip().split('\t')
                if len(pair) != 2:
                    print('ignore case in %s'%path, pair)
                    continue
                label = pair[1]
                line_data = pair[0]
                examples.append(data.Example.fromlist([line_data, label], fields))

        super(NlcDatasetSingleFile, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, path='.',
               train='train.txt', validation='dev.txt', test='test.txt', **kwargs):
        """Create dataset objects for splits of the SSTB dataset.
        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'dev.txt'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'test.txt'.
            train_subtrees: Whether to use all subtrees in the training set.
                Default: False.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        # path = cls.download_or_unzip(root)

        train_data = None if train is None else cls(True,
            os.path.join(path, train), text_field, label_field, **kwargs)
        val_data = None if validation is None else cls(False,
            os.path.join(path, validation), text_field, label_field, **kwargs)
        test_data = None if test is None else cls(False,
            os.path.join(path, test), text_field, label_field, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

