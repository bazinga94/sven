import os
import abc
import json
import torch
import random
from torch.utils.data import Dataset

from sven.constant import BINARY_LABELS, SEC_LABEL, VUL_LABEL, PROMPTS, CWES_TRAINED, CWES_TRAINED_SUBSET, ENPM_TRAINED
from sven.utils import get_indent

# Base Dataset class for processing training data
class DatasetBase(Dataset):
    def __init__(self, args, tokenizer, mode):
        self.args = args
        self.tokenizer = tokenizer
        self.dataset = list()
        if self.args.vul_type is not None:
            vul_types = [self.args.vul_type]
        else:
            # this is for incoder model
            if 'incoder' in self.args.pretrain_dir:
                vul_types = CWES_TRAINED_SUBSET
            else:
                # if we do not fix the code or args vul_types will be CWES_TRAINED
                # vul_types = CWES_TRAINED
                vul_types = ENPM_TRAINED

        # we might have only one vul_types
        # we should delete or change the i, vul_type
        # vul_type is only fo read json
        # @@@@@ but "i" is used in get_tensor() function 
        for i, vul_type in enumerate(vul_types):
            with open(os.path.join(args.data_dir, mode, f'{vul_type}.jsonl')) as f:
                lines = f.readlines()
            # each line is equal to one json object
            for line in lines:
                diff_j = json.loads(line)
                # if diff_j['file_name'].endswith('.py'):
                #     lang = 'py'
                # else:
                #     lang = 'c'

                # we only use python code
                lang = 'py'
                # we should adjust to our labels
                # labels = ["pandas", "numpy"]
                labels = [SEC_LABEL, VUL_LABEL]
                srcs = [diff_j['func_src_after'], diff_j['func_src_before']]
                if self.args.diff_level == 'prog':
                    diffs = [None, None]
                elif self.args.diff_level == 'line':
                    diffs = [diff_j['line_changes']['added'], diff_j['line_changes']['deleted']]
                elif self.args.diff_level == 'char':
                    diffs = [diff_j['char_changes']['added'], diff_j['char_changes']['deleted']]
                # default is mix
                # they adopt a mixing strategy
                # that utilizes character level masks for secure codes 
                # and line-level masks for unsafe codes(this is the most precise way? they said)
                # @@@@@ in our case, line or prog might be valid(let's check the diff from our data)
                elif self.args.diff_level == 'mix':
                    diffs = [diff_j['char_changes']['added'], diff_j['line_changes']['deleted']]
                else:
                    raise NotImplementedError()
                for label, src, changes in zip(labels, srcs, diffs):
                    self.add_data(label, src, changes, i, lang)

    @abc.abstractclassmethod
    def add_data(self, label, src, changes, vul_id):
        raise NotImplementedError()

    def __len__(self):
        return len(self.dataset)

    # call during an iteration of DatasetBase
    def __getitem__(self, item):
        return tuple(torch.tensor(t) for t in self.dataset[item])

# Dataset class for Prefix-based training
class PrefixDataset(DatasetBase):
    def __init__(self, args, tokenizer, mode):
        super().__init__(args, tokenizer, mode)

    # add jsonl data to dataset
    def add_data(self, label, src, changes, vul_id, lang):
        # SEC_LABEL = 'sec'
        # VUL_LABEL = 'vul'
        # BINARY_LABELS = [SEC_LABEL, VUL_LABEL]
        # secure code: 0 / vulnerable code: 1
        control_id = BINARY_LABELS.index(label)    
        data = self.get_tensor(src, vul_id, control_id, changes)    # vul_id = cwe-xx index
        # there is no usage or changed in get_tensor()
        # vul_id directly append to dataset
        if data is not None:
            self.dataset.append(data)

    def get_tensor(self, src, vul_id, control_id, changes):
        # encoding part
        # tokenize the source code and transform it to dict
        be = self.tokenizer.encode_plus(src)
        # token ID that are used as input of model
        tokens = be.data['input_ids']
        # check token length?
        if len(tokens) > self.args.max_num_tokens: return None

        # @@@@@ ??, can't find info from the project's code about cwe-invalid, cwe-valid
        min_changed_tokens = (2 if self.args.vul_type in ('cwe-invalid', 'cwe-valid') else 1)
        # if diff_level == 'prog':
        if changes is None:
            # All tokens are considered security-sensitive and are masked with 1.
            weights = [1] * len(tokens)
        else:
            # line, char, mix
            weights = [0] * len(tokens)
            for change in changes:
                char_start = change['char_start']
                char_start_idx = be.char_to_token(char_start)
                char_end = change['char_end']
                char_end_idx = be.char_to_token(char_end-1)
                # convert the start and end of that string to the token index used by the tokenizer
                for char_idx in range(char_start_idx, char_end_idx+1):
                    # Set the weight to 1 for tokens in the changed part.
                    weights[char_idx] = 1
            # if number of changed tokens are too many or too few, return None
            if sum(weights) < min_changed_tokens: return None
            if len(tokens) - sum(weights) < min_changed_tokens: return None

        return tokens, weights, control_id, vul_id

# Dataset class for Prompt-based training -> We might not use this!!!!!
class TextPromptDataset(DatasetBase):
    def __init__(self, args, tokenizer, mode):
        super().__init__(args, tokenizer, mode)

    def add_data(self, label, src, changes, vul_id, lang):
        control_id = BINARY_LABELS.index(label)    
        if lang == 'py':
            control = get_indent(src) + '# ' + PROMPTS[control_id]
        else:
            control = get_indent(src) + '// ' + PROMPTS[control_id]
        src = control + src
        data = self.get_tensor(src, control, changes)
        if data is not None:
            self.dataset.append(data)

    def get_tensor(self, src, control, changes):
        be = self.tokenizer.encode_plus(src)
        tokens = be.data['input_ids']

        if changes is None:
            labels = tokens[:]
        else:
            labels = [-100] * len(tokens)
            label_set = False
            for change in changes:
                char_start = change['char_start'] + len(control)
                char_start_idx = be.char_to_token(char_start)
                char_end = change['char_end'] + len(control)
                char_end_idx = be.char_to_token(char_end-1)
                for i in range(char_start_idx, char_end_idx+1):
                    labels[i] = tokens[i]
                    label_set = True
            if not label_set: return None

        if len(tokens) > self.args.max_num_tokens: return None
        return tokens, labels