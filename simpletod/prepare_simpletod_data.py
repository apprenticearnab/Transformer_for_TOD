from utils.args_parser import ArgsParser
from data.dataset.multiwoz import MultiWozDataset
#import en_core_web_sm
from nltk import ngrams
from utils.multiwoz import dbPointer
import ipdb
import json
import random
import os

from transformers import GPT2Tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

multiwoz_data = json.load(open('Processed_data/multi-woz/lex.json', 'r'))
save_dir = 'Processed_data/simptod'
os.makedirs(save_dir, exist_ok=True)
f1_train = open('Processed_data/simptod/context_train.txt','w')
f1_val = open('Processed_data/simptod/context_val.txt','w')
f1_test = open('Processed_data/simptod/context_test.txt','w')
f2_train = open('Processed_data/simptod/belief_train.txt','w')
f2_val = open('Processed_data/simptod/belief_val.txt','w')
f2_test = open('Processed_data/simptod/belief_test.txt','w')
f3_train = open('Processed_data/simptod/action_train.txt','w')
f3_val = open('Processed_data/simptod/action_val.txt','w')
f3_test = open('Processed_data/simptod/action_test.txt','w')


for split in ['train', 'val', 'test']:

    opt = ArgsParser().parse()
    opt.use_knowledge = True
    opt.use_action = True
    opt.context_knowledge = True
    opt.lexical = True

    data = MultiWozDataset(opt, split=split, shuffle=False)

    opt_delex = ArgsParser().parse()
    data_delex = MultiWozDataset(opt_delex, split=split, shuffle=False)

    history_raw_new = []
    belief_raw_new = []
    belief_raw_none_new = []
    action_raw_new = []
    output_raw_new = []
    output_raw_delex_new = []
    db_search_raw = []
    db_nmatch_raw = []
    
    if split == 'test':
        test_dict = {}

    lex_dict = {}
    delex_dict = {}
    for d in data:
        lex_dict[d['name']] = d

    for d in data_delex:
        delex_dict[d['name']] = d

    for key in lex_dict:
        d_lex = lex_dict[key]
        d_delex = delex_dict[key]
        inp = d_lex['input_raw']
        out = d_lex['target_raw']
        out_delex = d_delex['target_raw']
        db_data = d_lex['db']
        goal = multiwoz_data[key]['goal']
        
        #print(inp[1])
        for i, (usr, sys) in enumerate(zip(inp, out)):
            if i == 0:
                history_new = '<|context|> <|user|> {} <|endofcontext|>'.format(usr)
            else:
                tmp_new = ['<|context|>']
                for k in range(i):

                    tmp_new.append('<|user|> ' + inp[k])
                    tmp_new.append('<|system|> ' + out[k])

                tmp_new.append('<|user|> ' + usr + '<|endofcontext|>')
                history_new = ' '.join(tmp_new)
            #print(history_new)
            if split == 'train':
                f1_train.write('{}\n'.format(history_new))
            elif split=='val':
                f1_val.write('{}\n'.format(history_new))
            else:
                f1_test.write('{}\n'.format(history_new))
            sys_delex = out_delex[i]
            history_raw_new.append(history_new)
            output_raw_new.append('<|response|> ' + sys + ' <|endofresponse|>')

            output_raw_delex_new.append('<|response|> ' + sys_delex.strip() + ' <|endofresponse|>')

            db_text = dbPointer.convert_dbpointer_to_text(db_data[i], goal, d_lex['belief_raw'][i])
            db_search_raw.append('<|dbsearch|> {} <|endofdbsearch|>'.format(db_text))

            db_text_nmatch = dbPointer.convert_dbpointer_to_text_nmatch(db_data[i], goal, d_lex['belief_raw'][i])
            db_nmatch_raw.append('<|dbsearch|> {} <|endofdbsearch|>'.format(db_text_nmatch))

        belief = d_lex['belief_raw']
        #print(belief)
        for bs in belief:
            tmp_bs_new = []
            for i, b in enumerate(bs):
                if b[-1] in ['not mentioned']: # comment this for DST task
                    continue
                if i == len(bs) - 1:
                    tmp_bs_new.append(' '.join(b))
                else:
                    tmp_bs_new.append(' '.join(b))

            if len(tmp_bs_new) == 0:
                tmp_bs_new.append(' ')

            tmp_new = '<|belief|> {} <|endofbelief|>'.format(' , '.join(tmp_bs_new))
            belief_raw_new.append(tmp_new)
            if split == 'train':
                f2_train.write('{}\n'.format(tmp_new))
            elif split=='val':
                f2_val.write('{}\n'.format(tmp_new))
            else:
                f2_test.write('{}\n'.format(tmp_new))
        # belief for DST task (include none)
        for bs in belief:
            tmp_bs_new = []
            for i, b in enumerate(bs):
                if i == len(bs) - 1:
                    tmp_bs_new.append(' '.join(b))
                else:
                    tmp_bs_new.append(' '.join(b))

            if len(tmp_bs_new) == 0:
                tmp_bs_new.append(' ')

            tmp_new = '<|belief|> {} <|endofbelief|>'.format(' , '.join(tmp_bs_new))
            belief_raw_none_new.append(tmp_new)
        #print(tmp_new)
        action = d_lex['action_raw']
        for act in action:
            tmp_act_new = []
            for i, a in enumerate(act):
                if i == len(act) - 1:
                    tmp_act_new.append(' '.join(a))
                else:
                    tmp_act_new.append(' '.join(a))
            if len(tmp_act_new) == 0:
                tmp_act_new.append(' ')

            tmp_new = '<|action|> {} <|endofaction|>'.format(' , '.join(tmp_act_new))
            action_raw_new.append(tmp_new)
            if split == 'train':
                f3_train.write('{}\n'.format(tmp_new))
            elif split=='val':
                f3_val.write('{}\n'.format(tmp_new))
            else:
                f3_test.write('{}\n'.format(tmp_new))

    tmp = []
    for inp, bs, dbsearch, act, trg in zip(history_raw_new, belief_raw_new, db_search_raw, action_raw_new, output_raw_delex_new):
        tmp.append(' '.join([inp.lower(), bs.lower(), dbsearch.lower(), act, trg]))
    with open('{}/{}.history_belief_dbsearch_action_sys_delex'.format(save_dir, split), 'wt') as f:
        for l in tmp:
            f.write('{} {}\n'.format(gpt2_tokenizer._bos_token, l.lower()))

    tmp = []
    for inp, bs, dbsearch, act, trg in zip(history_raw_new, belief_raw_new, db_nmatch_raw, action_raw_new,
                                           output_raw_delex_new):
        tmp.append(' '.join([inp.lower(), bs.lower(), dbsearch.lower(), act, trg]))
    with open('{}/{}.history_belief_dbnmatch_action_sys_delex'.format(save_dir, split), 'wt') as f:
        for l in tmp:
            f.write('{} {}\n'.format(gpt2_tokenizer._bos_token, l.lower()))

    with open('{}/{}.history'.format(save_dir, split), 'wt') as f:
        for l in history_raw_new:
            f.write('{} {}\n'.format(gpt2_tokenizer._bos_token, l.lower()))

    tmp = []
    for hist, bs in zip(history_raw_new, belief_raw_none_new):
        tmp.append(' '.join([hist.lower(), bs.lower()]))
    with open('{}/{}.history_belief'.format(save_dir, split),
              'wt') as f:
        for l in tmp:
            f.write('{} {} {}\n'.format(gpt2_tokenizer._bos_token, l.lower(), gpt2_tokenizer._eos_token))

    tmp = []
    for hist, bs, act, trg in zip(history_raw_new, belief_raw_new, action_raw_new, output_raw_delex_new):
        tmp.append(' '.join([hist.lower(), bs.lower(), act, trg]))
    with open('{}/{}.history_belief_action_sys_delex'.format(save_dir, split), 'wt') as f:
        for l in tmp:
            f.write('{} {} {}\n'.format(gpt2_tokenizer._bos_token, l.lower(), gpt2_tokenizer._eos_token))
           
        
f1_train.close()
f1_val.close()
f1_test.close()
f2_train.close()
f2_val.close()
f2_test.close()
f3_train.close()
f3_val.close()
f3_test.close()