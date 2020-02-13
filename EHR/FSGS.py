import time
import torch
import torch.nn as nn
from ThreeAttackAlgorithms import rnn_tools
from ThreeAttackAlgorithms import rnn_model
from torch.autograd import Variable
from copy import deepcopy
import os
import pickle
import numpy as np
import itertools
from math import exp

TAU = 0.7
MODEL_TYPE = 'sub'
SECONDS = 3600

TITLE = '=== ' + MODEL_TYPE + ' target prob = ' + str(TAU) + ' time = ' + str(SECONDS) + ' ==='

NUM_ATTACK_SAMPLES = 500

log_f = open('./Logs/Prim_%s_t=%s_s=%d.bak' % (MODEL_TYPE, str(TAU), SECONDS), 'a')


class Attacker(object):
    def __init__(self, options, emb_weights):
        print("Loading pre-trained classifier...", file=log_f, flush=True)

        self.model = rnn_model.LSTM(options, emb_weights)

        if MODEL_TYPE == 'sub':
            self.model.load_state_dict(torch.load('./Classifiers/Submodular_lstm.49')) # abs
        elif MODEL_TYPE == 'nonsub':
            self.model.load_state_dict(torch.load('./Classifiers/Nonsubmodular_lstm.42')) # positive and negative
        self.model.eval()

        self.criterion = nn.CrossEntropyLoss()

    def classify(self, person):

        model_input, weight_of_embed_codes = self.input_handle(person)

        logit = self.forward_lstm(model_input, weight_of_embed_codes)

        pred = torch.max(logit, 1)[1].view((1,)).data.numpy()

        prob = logit[0][pred]

        return pred, prob

    def input_handle(self, person):
        t_diagnosis_codes = rnn_tools.pad_matrix(person)
        model_input = deepcopy(t_diagnosis_codes)
        for i in range(len(model_input)):
            for j in range(len(model_input[i])):
                idx = 0
                for k in range(len(model_input[i][j])):
                    model_input[i][j][k] = idx
                    idx += 1

        model_input = Variable(torch.LongTensor(model_input))
        return model_input.transpose(0, 1), torch.tensor(t_diagnosis_codes).transpose(0, 1)

    def forward_lstm(self, model_input, weight_of_embed_codes):
        x = torch.LongTensor(model_input)
        x = self.model.embed(x)
        weight_of_embed_codes = torch.unsqueeze(weight_of_embed_codes, dim=3)
        x = x * weight_of_embed_codes
        x = self.model.relu(x)
        x = torch.mean(x, dim=2)
        # 1
        # h0 = Variable(torch.FloatTensor(torch.zeros(1, x.size()[1], x.size()[2])))
        # c0 = Variable(torch.FloatTensor(torch.zeros(1, x.size()[1], x.size()[2])))
        # output, h_n = self.model.lstm(x, (h0, c0))
        # embedding, attn_weights = self.model.attention(output.transpose(0, 1))
        # logit = self.model.fc(embedding)
        # logit = self.model.softmax(logit)
        # 2
        h0 = Variable(torch.FloatTensor(torch.randn(1, x.size()[1], x.size()[2])))
        c0 = Variable(torch.FloatTensor(torch.randn(1, x.size()[1], x.size()[2])))
        output, h_n = self.model.lstm(x, (h0, c0))
        embedding, attn_weights = self.model.attention(output.transpose(0, 1))
        x = self.model.dropout(embedding)
        logit = self.model.fc(x)
        logit = self.model.softmax(logit)
        return logit

    def eval_Cy(self, person, pos, greedy_set, greedy_set_max_Cy, target_label):
        best_Cy = greedy_set_max_Cy
        best_temp_person = deepcopy(person)
        candidate_lists = []

        # candidate_lists contains all the non-empty subsets of greedy_set
        if greedy_set:
            for i in range(1, len(greedy_set) + 1):
                subset1 = itertools.combinations(greedy_set, i)
                for subset in subset1:
                    candidate_lists.append(list(subset))

            for can in candidate_lists:
                can.append(pos)
        else:
            candidate_lists.append([pos])

        for can in candidate_lists:

            temp_person = deepcopy(person)

            for position in can:
                visit_idx = position[0]
                code_idx = position[1]
                if code_idx in person[visit_idx]:
                    temp_person[visit_idx].remove(code_idx)
                else:
                    temp_person[visit_idx].append(code_idx)

            model_input, weight_of_embed_codes = self.input_handle(temp_person)

            logit = self.forward_lstm(model_input, weight_of_embed_codes)

            Cy = logit[0][target_label]

            if Cy > best_Cy:
                best_Cy = Cy
                best_temp_person = deepcopy(temp_person)

        return best_Cy, best_temp_person

    def attack(self, person, y):
        st = time.time()
        success_flag = 1

        one_hot_codes = np.zeros((len(person), 4130))
        for visit_idx in range(len(person)):
            for code in person[visit_idx]:
                one_hot_codes[visit_idx][code] = 1

        orig_pred, orig_prob = self.classify(person)
        print(orig_pred, orig_prob, file=log_f, flush=True)

        pred, pred_prob = orig_pred, orig_prob

        if not (pred == y or pred_prob < TAU):
            success_flag = 0
            return {}, person, pred_prob, 0, success_flag, 0

        best_Cy = 1 - pred_prob

        iteration = 0

        n_change = 0

        greedy_set = set()

        greedy_set_max_Cy = best_Cy

        while (pred == y or best_Cy < TAU):
            best_pos = 0
            iteration += 1

            best_temp_Cy = -1

            for visit_idx in range(len(one_hot_codes)):
                for code_idx in range(len(one_hot_codes[visit_idx])):

                    if (time.time() - st) > SECONDS and (pred == y or pred_prob < TAU):
                        print("===== Time out! Attack Fail =====", file=log_f, flush=True)
                        success_flag = -1
                        break

                    pos = (visit_idx, code_idx)
                    if pos in greedy_set: continue
                    eval_person = deepcopy(person)
                    Cy, temp_person = self.eval_Cy(eval_person, pos, greedy_set, greedy_set_max_Cy, 1-y)

                    if Cy > best_temp_Cy:
                        best_temp_Cy = Cy
                        best_pos = pos
                        if best_temp_Cy > best_Cy:
                            best_Cy = best_temp_Cy
                            best_temp_person = deepcopy(temp_person)

                if (time.time() - st) > SECONDS and (pred == y or pred_prob < TAU):
                    print("===== Time out! Attack Fail =====", file=log_f, flush=True)
                    success_flag = -1
                    break

            if best_pos:
                greedy_set.add(best_pos)
                greedy_set_max_Cy = best_Cy
                greedy_set_best_temp_person = best_temp_person
            # else:
            #     success_flag = 0
            #     print("====== Fail ======")
            #     break

            if best_Cy > 0.5:
                pred = 1 - y

            # if (time.time() - st) > SECONDS and (pred == y or pred_prob < TAU):
            #     print("=====Fail: Time out! Attack failed.=====", file=log_f, flush=True)
            #     success_flag = 0
            #     break

        flag = 0
        for v_idx in range(len(greedy_set_best_temp_person)):
            for code in person[v_idx]:
                if code not in greedy_set_best_temp_person[v_idx]:
                    flag = 1
                    n_change += 1
            for code in greedy_set_best_temp_person[v_idx]:
                if code not in person[v_idx]:
                    n_change += 1
        if flag:
            print("=== Some original codes is reduced (from 1 to 0). ===", file=log_f, flush=True)

        return greedy_set, greedy_set_best_temp_person, best_Cy, n_change, success_flag, iteration


def main(emb_weights, training_file, validation_file,
                                       testing_file, n_diagnosis_codes, n_labels,
                                       batch_size, dropout_rate,
                                       L2_reg, n_epoch, log_eps, n_claims, visit_size, hidden_size,
                                       use_gpu, model_name):
    options = locals().copy()
    print("Loading dataset...", file=log_f, flush=True)
    test = rnn_tools.load_data(training_file, validation_file, testing_file)

    n_people = NUM_ATTACK_SAMPLES

    attacker = Attacker(options, emb_weights)

    n_success = 89

    total_node_change = n_success*1.101123595505618

    n_iteration = n_success*1.101123595505618

    saving_time = {}

    attack_code_dict = {}

    # ****** #
    file1 = './AttackCodes/codes_Prim_sub_t=0.7_s=3600.pickle'

    attack_code_dict = pickle.load(open(file1, 'rb'))

    file2 = './Each_TimeNumber/time_Prim_sub_t=0.7_s=3600.pickle'

    saving_time = pickle.load(open(file2, 'rb'))
    # ****** #

    for i in range(100, n_people):
        print("-------- %d ---------" % (i), file=log_f, flush=True)

        person = test[0][i]

        y = test[1][i]

        n_visit = len(person)

        print('* Processing:%d/%d person, number of visit for this person: %d' % (i, n_people, n_visit), file=log_f, flush=True)

        print("* Original: "+str(person), file=log_f, flush=True)

        print("  Original label: %d" % (y), file=log_f, flush=True)

        st = time.time()
        greedy_set, greedy_set_best_temp_person, best_Cy, num_changed, success_flag, iteration = attacker.attack(person, y)

        et = time.time()
        all_t = et-st

        if success_flag:
            n_success += 1
            n_iteration += iteration
            total_node_change += num_changed

            saving_time[i] = all_t
            attack_code_dict[i] = list(greedy_set)

        print("* Result: ", file=log_f, flush=True)
        print(greedy_set, file=log_f, flush=True)
        print(greedy_set_best_temp_person, file=log_f, flush=True)
        print(best_Cy, file=log_f, flush=True)

        print("  Nnumber of changed codes: %d" % (num_changed), file=log_f, flush=True)

        print("  Number of iterations for this: " + str(iteration), file=log_f, flush=True)

        print(" Time: "+str(all_t), file=log_f, flush=True)

        print("* SUCCESS Number NOW: %d " % (n_success), file=log_f, flush=True)

        if n_success:
            print("  Average Number of success changed codes: " + str(float(total_node_change)/float(n_success)), file=log_f, flush=True)
            print("  Average Number of success iterations: " + str(float(n_iteration) / float(n_success)), file=log_f, flush=True)

            if i % 20 == 0:
                pickle.dump(attack_code_dict, open('./AttackCodes/codes_Prim_%s_t=%s_s=%d.pickle' % (MODEL_TYPE, str(TAU), SECONDS), 'wb'))
                pickle.dump(saving_time, open('./Each_TimeNumber/time_Prim_%s_t=%s_s=%d.pickle' % (MODEL_TYPE, str(TAU), SECONDS), 'wb'))

        pickle.dump(attack_code_dict, open('./AttackCodes/codes_Prim_%s_t=%s_s=%d.pickle' % (MODEL_TYPE, str(TAU), SECONDS), 'wb'))
        pickle.dump(saving_time, open('./Each_TimeNumber/time_Prim_%s_t=%s_s=%d.pickle' % (MODEL_TYPE, str(TAU), SECONDS), 'wb'))

    print("--- Total Success Number: " + str(n_success) + " ---", file=log_f, flush=True)
    print(TITLE)
    print(TITLE, file=log_f, flush=True)



if __name__ == '__main__':
    print(TITLE, file=log_f, flush=True)
    print(TITLE)
    # parameters
    batch_size = 5
    dropout_rate = 0.5
    L2_reg = 0.001  # 0.001
    log_eps = 1e-8
    n_epoch = 50
    n_labels = 2  # binary classification
    visit_size = 70
    hidden_size = 70
    n_diagnosis_codes = 4130
    n_claims = 504

    use_gpu = False
    model_name = 'lstm'

    trianing_file = './SourceData/hf_dataset_training.pickle'
    validation_file = './SourceData/hf_dataset_validation.pickle'
    testing_file = './SourceData/hf_dataset_testing.pickle'

    emb_weights_char = torch.load("./SourceData/PretrainedEmbedding.4")['char_embeddings.weight']
    emb_weights_word = torch.load("./SourceData/PretrainedEmbedding.4")['word_embeddings.weight']

    ##################

    map_char_idx = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '.': 10, 'E': 11,
                    'V': 12, 'VAC': 13}

    tree = pickle.load(open('./SourceData/hf_dataset_270_code_dict.pickle', 'rb'))
    map_codeidx_charidx = {}

    for k in tree.keys():
        codeidx = tree[k]
        charidx = []

        code = str(k)
        len_code = len(code)

        if len_code == 7:
            for c in code:
                charidx.append(map_char_idx[c])

        elif len_code == 6:

            if code[0] == 'V':
                charidx.append(map_char_idx[code[0]])
                charidx.append(map_char_idx['VAC'])
                for i in range(1, 6):
                    charidx.append(map_char_idx[code[i]])

            elif code[0] == 'E':
                charidx.append(map_char_idx[code[0]])
                for i in range(1, 6):
                    charidx.append(map_char_idx[code[i]])
                charidx.append(map_char_idx['VAC'])

            else:
                charidx.append(map_char_idx['VAC'])
                for i in range(6):
                    charidx.append(map_char_idx[code[i]])

        elif len_code == 5:

            if code[0] == 'V':
                charidx.append(map_char_idx[code[0]])
                charidx.append(map_char_idx['VAC'])
                for i in range(1, 5):
                    charidx.append(map_char_idx[code[i]])
                charidx.append(map_char_idx['VAC'])

            else:
                charidx.append(map_char_idx['VAC'])
                for i in range(5):
                    charidx.append(map_char_idx[code[i]])
                charidx.append(map_char_idx['VAC'])

        elif len_code == 4:
            for i in range(4):
                charidx.append(map_char_idx[code[i]])
            charidx.append(map_char_idx['VAC'])
            charidx.append(map_char_idx['VAC'])
            charidx.append(map_char_idx['VAC'])

        elif len_code == 3:
            if code[0] == 'V':
                charidx.append(map_char_idx[code[0]])
                charidx.append(map_char_idx['VAC'])
                charidx.append(map_char_idx[code[1]])
                charidx.append(map_char_idx[code[2]])
                charidx.append(map_char_idx['VAC'])
                charidx.append(map_char_idx['VAC'])
                charidx.append(map_char_idx['VAC'])
            else:
                charidx.append(map_char_idx['VAC'])
                charidx.append(map_char_idx[code[0]])
                charidx.append(map_char_idx[code[1]])
                charidx.append(map_char_idx[code[2]])
                charidx.append(map_char_idx['VAC'])
                charidx.append(map_char_idx['VAC'])
                charidx.append(map_char_idx['VAC'])

        map_codeidx_charidx[codeidx] = charidx

    codes_embedding = []

    for i in range(4130):
        chars = map_codeidx_charidx[i]

        char_code_embedding = []
        for c in chars:
            c_embedding = emb_weights_char[c].tolist()
            char_code_embedding.append(c_embedding)

        # char_code_embedding = np.mean(char_code_embedding, axis=0)
        char_code_embedding = np.reshape(char_code_embedding, (-1))

        word_embedding = np.array(emb_weights_word[i])

        # code_embedding = 0.5 * (char_code_embedding + word_embedding)
        code_embedding = 0.5 * char_code_embedding + 0.5 * word_embedding

        codes_embedding.append(code_embedding)
    ##################

    emb_weights = torch.tensor(codes_embedding, dtype=torch.float)
    main(emb_weights, trianing_file, validation_file,
                                       testing_file, n_diagnosis_codes, n_labels,
                                       batch_size, dropout_rate,
                                       L2_reg, n_epoch, log_eps, n_claims, visit_size, hidden_size,
                                       use_gpu, model_name)