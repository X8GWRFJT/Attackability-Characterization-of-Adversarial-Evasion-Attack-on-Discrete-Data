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
from math import pow
from itertools import combinations

TAU = 0.7
INDICE_K = 4
MODEL_TYPE = 'nonsub'
SECONDS = 180

TITLE = '=== ' + MODEL_TYPE + ' target prob = ' + str(TAU) + ' k = ' \
        + str(INDICE_K) + ' time = ' + str(SECONDS) + ' ==='

NUM_ATTACK_SAMPLES = 500

log_f = open('./Logs/GB_%s_k=%d_t=%s_s=%d.bak' % (MODEL_TYPE, INDICE_K, str(TAU), SECONDS), 'w+')


class Attacker(object):
    def __init__(self, options, emb_weights):
        print("Loading pre-trained classifier...", file=log_f, flush=True)

        self.model = rnn_model.LSTM(options, emb_weights)
        self.model2 = rnn_model.LSTM(options, emb_weights)

        if MODEL_TYPE == 'sub':
            self.model.load_state_dict(torch.load('./Classifiers/Submodular_lstm.49')) # abs
            self.model2.load_state_dict(torch.load('./Classifiers/Submodular_lstm.49'))
        elif MODEL_TYPE == 'nonsub':
            self.model.load_state_dict(torch.load('./Classifiers/Nonsubmodular_lstm.42')) # positive and negative
            self.model2.load_state_dict(torch.load('./Classifiers/Nonsubmodular_lstm.42'))

        self.model.eval()
        self.model2.eval()

        self.criterion = nn.CrossEntropyLoss()

    def classify(self, person, model):

        model_input, weight_of_embed_codes = self.input_handle(person)

        logit = self.model2(model_input, weight_of_embed_codes)

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

    def forward_lstm(self, weighted_embed_codes, model):
        x = model.relu(weighted_embed_codes)
        x = torch.mean(x, dim=2)
        h0 = Variable(torch.FloatTensor(torch.randn(1, x.size()[1], x.size()[2])))
        c0 = Variable(torch.FloatTensor(torch.randn(1, x.size()[1], x.size()[2])))
        output, h_n = model.lstm(x, (h0, c0))
        embedding, attn_weights = model.attention(output.transpose(0, 1))
        x = model.dropout(embedding)  # (n_samples, hidden_size)

        logit = model.fc(x)  # (n_samples, n_labels)

        logit = model.softmax(logit)
        return logit

    def attack(self, person, y):
        st = time.time()

        success_flag = 1

        one_hot_codes = np.zeros((len(person), 4130))
        for visit_idx in range(len(person)):
            for code in person[visit_idx]:
                one_hot_codes[visit_idx][code] = 1

        orig_pred, orig_prob = self.classify(person, self.model)
        print(orig_pred, orig_prob, file=log_f, flush=True)

        pred, pred_prob = orig_pred, orig_prob

        if not (pred == y or pred_prob < TAU):
            success_flag = 0
            return pred_prob, {}, {}, success_flag, 0

        maxCy = 1 - pred_prob

        S_t = set()

        new_code = (-1,-1)
        Code_S_t = []

        iteration = 0

        n_change = 0

        first_time_flag = 1

        while (pred == y or maxCy < TAU):

            iteration += 1

            persons = []

            flip_num = int(pow(2, len(S_t)))

            selected_codes = []

            if first_time_flag:
                flip_num = 1
                persons.append(person)
                first_time_flag = 0
            else:
                for i in range(len(S_t) + 1):
                    comb_set = combinations(list(S_t), i)
                    for s in comb_set:
                        selected_codes.append(list(s))
                if selected_codes:
                    for c in selected_codes:
                        c.append(new_code)
                else:
                    selected_codes.append([new_code])

                for c in selected_codes: # [[C], [A, C], [B, C], [A, B, C]]
                    temp_person = deepcopy(person)
                    for code in c:
                        if code[1] in person[code[0]]:
                            temp_person[code[0]].remove(code[1])
                        else:
                            temp_person[code[0]].append(code[1])
                    persons.append(temp_person)

                S_t.add(new_code)

            temp_vs = []
            temp_Cys = []
            map_temp_v_subset = {}

            for i in range(flip_num):
                # model_input, weight_of_embed_codes = self.input_handle(persons[i])
                # embed_codes = self.model.embed(torch.LongTensor(model_input))
                #
                # weight_of_embed_codes = Variable(weight_of_embed_codes.data, requires_grad=True)
                #
                # weighted_embed_codes = embed_codes * torch.unsqueeze(weight_of_embed_codes, dim=3)
                #
                # output = self.forward_lstm(weighted_embed_codes, self.model)
                #
                # loss = self.criterion(output, Variable(torch.LongTensor([y])))
                #
                # loss.backward()
                #
                # score = np.zeros((len(person), 4130))
                #
                # for visit_idx in range(len(one_hot_codes)):
                #     for code_idx in range(len(one_hot_codes[visit_idx])):
                #         pos = (visit_idx, code_idx)
                #         if pos in list(S_t):
                #             score[visit_idx][code_idx] = 0
                #             continue
                #         a = weight_of_embed_codes.grad.data[visit_idx][0][code_idx]
                #
                #         score[visit_idx][code_idx] = abs(a.data)
                #
                # indices = np.argsort(-np.reshape(score, (-1)))[:INDICE_K]
                indices = np.random.choice()

                temp_v, temp_Cy = self.code_paraphrase(persons[i], indices, y)
                temp_vs.append(temp_v)
                temp_Cys.append(temp_Cy)
                map_temp_v_subset[temp_v] = i

            best_temp_Cy_idx = np.argmax(temp_Cys)
            best_temp_Cy = np.max(temp_Cys)
            new_code = temp_vs[best_temp_Cy_idx]

            if best_temp_Cy > maxCy:
                maxCy = best_temp_Cy
                if selected_codes:
                    Code_S_t = selected_codes[map_temp_v_subset[new_code]]
                    Code_S_t.append(new_code)
                else:
                    Code_S_t = [new_code]

            if best_temp_Cy > 0.5:
                pred = 1-y

            if (time.time() - st) > SECONDS and (pred == y or pred_prob < TAU):
                print("===== Time out! Attack Fail =====", file=log_f, flush=True)
                success_flag = -1
                break

        S_t.add(new_code)

        return maxCy, S_t, Code_S_t, success_flag, iteration

    def code_paraphrase(self, person, indices, y):
        temp_v_index = -1
        temp_Cy = -1
        for pos in indices:
            visit_idx = pos // 4130
            code_idx = pos % 4130
            candidate = deepcopy(person)
            if code_idx in person[visit_idx]:
                candidate[visit_idx].remove(code_idx)
            else:
                candidate[visit_idx].append(code_idx)
            model_input, weight_of_embed_codes = self.input_handle(candidate)
            pred_prob = self.model(model_input, weight_of_embed_codes)
            if pred_prob[0][1-y] > temp_Cy:
                temp_Cy = pred_prob[0][1-y]
                temp_v_index = pos
        return (temp_v_index//4130, temp_v_index%4130), temp_Cy


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

    n_success = 0
    n_fail = 0

    total_node_change = 0

    n_iteration = 0

    saving_time = {}

    attack_code_dict = {}

    for i in range(n_people):
        print("-------- %d ---------" % (i), file=log_f, flush=True)

        person = test[0][i]

        y = test[1][i]

        n_visit = len(person)

        print('* Processing:%d/%d person, number of visit for this person: %d' % (i, n_people, n_visit), file=log_f, flush=True)

        print("* Original: "+str(person), file=log_f, flush=True)

        print("  Original label: %d" % (y), file=log_f, flush=True)

        st = time.time()
        # changed_person, score, num_changed, success_flag, iteration, changed_pos = attacker.attack(person, y)
        maxCy, S_t, Code_S_t, success_flag, iteration = attacker.attack(person, y)

        et = time.time()
        all_t = et-st

        if success_flag == 1:
            n_success += 1
            n_iteration += iteration
            total_node_change += len(Code_S_t)

            saving_time[i] = all_t
            attack_code_dict[i] = Code_S_t
        elif success_flag == -1:
            n_fail += 1

        print("* Result: ", file=log_f, flush=True)

        print(Code_S_t, file=log_f, flush=True)
        print(S_t, file=log_f, flush=True)
        print(maxCy, file=log_f, flush=True)

        print("  Nnumber of changed codes: %d" % (len(Code_S_t)), file=log_f, flush=True)

        print("  Number of iterations for this: " + str(iteration), file=log_f, flush=True)

        print(" Time: "+str(all_t), file=log_f, flush=True)

        print("* SUCCESS Number NOW: %d " % (n_success), file=log_f, flush=True)
        print("* Failure Number NOW: %d " % (n_fail), file=log_f, flush=True)

        if n_success:
            print("  Average Number of success changed codes: " + str(float(total_node_change)/float(n_success)), file=log_f, flush=True)
            print("  Average Number of success iterations: " + str(float(n_iteration) / float(n_success)), file=log_f, flush=True)

        if i % 20 == 0:
            pickle.dump(attack_code_dict, open('./AttackCodes/codes_GB_%s_k=%d_t=%s_s=%d.pickle' % (MODEL_TYPE, INDICE_K, str(TAU), SECONDS), 'wb'))
            pickle.dump(saving_time, open('Each_TimeNumber/time_GB_%s_k=%d_t=%s_s=%d.pickle' % (MODEL_TYPE, INDICE_K, str(TAU), SECONDS), 'wb'))

    pickle.dump(attack_code_dict,
                open('./AttackCodes/codes_GB_%s_k=%d_t=%s_s=%d.pickle' % (MODEL_TYPE, INDICE_K, str(TAU), SECONDS), 'wb'))
    pickle.dump(saving_time,
                open('Each_TimeNumber/time_GB_%s_k=%d_t=%s_s=%d.pickle' % (MODEL_TYPE, INDICE_K, str(TAU), SECONDS), 'wb'))

    print("--- Total Success Number: " + str(n_success) + " ---", file=log_f, flush=True)
    print("--- Total Attack Success Rate: " + str(float(n_success) / float(n_success + n_fail)), file=log_f, flush=True)

    print(TITLE, file=log_f, flush=True)
    print(TITLE)


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

        char_code_embedding = np.reshape(char_code_embedding, (-1))

        word_embedding = np.array(emb_weights_word[i])

        code_embedding = 0.5 * char_code_embedding + 0.5 * word_embedding

        codes_embedding.append(code_embedding)
    ##################

    emb_weights = torch.tensor(codes_embedding, dtype=torch.float)
    main(emb_weights, trianing_file, validation_file,
                                       testing_file, n_diagnosis_codes, n_labels,
                                       batch_size, dropout_rate,
                                       L2_reg, n_epoch, log_eps, n_claims, visit_size, hidden_size,
                                       use_gpu, model_name)