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
from math import exp

TAU = 0.5
INDICE_K = 10
MODEL_TYPE = 'sub'
SECONDS = 180

TITLE = '=== ' + MODEL_TYPE + ' target prob = ' + str(TAU) + ' k = ' \
        + str(INDICE_K) + ' time = ' + str(SECONDS) + ' ==='

NUM_ATTACK_SAMPLES = 500

log_f = open('./Logs/QL_%s_k=%d_t=%s_s=%d.bak' % (MODEL_TYPE, INDICE_K, str(TAU), SECONDS), 'a')


class Attacker(object):
    def __init__(self, options, emb_weights):
        print("Loading pre-trained classifier...", file=log_f, flush=True)

        self.model = rnn_model.LSTM(options, emb_weights)
        self.model2 = rnn_model.LSTM(options, emb_weights)

        if MODEL_TYPE == 'sub':
            self.model.load_state_dict(torch.load('./Classifiers/Submodular_lstm.49')) # abs
            self.model2.load_state_dict(torch.load('./Classifiers/Submodular_lstm.49'))
        elif MODEL_TYPE == 'nonsub':
            self.model.load_state_dict(torch.load('./Classifiers/Nonsubmodular_lstm.42'))  # positive and negative
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

    def code_paraphrase(self, one_hot_codes, person, indices, y):
        candidates = [person]
        for pos in indices:
            visit_idx = pos // 4130
            code_idx = pos % 4130
            currant_candidate = deepcopy(candidates)

            if one_hot_codes[visit_idx][code_idx] == 0:
                repl = 1
            else:
                repl = 0

            for c in candidates:
                corrupted = deepcopy(c)
                if repl == 1:
                    corrupted[visit_idx].append(code_idx)
                else:
                    if code_idx in corrupted[visit_idx]:
                        corrupted[visit_idx].remove(code_idx)
                currant_candidate.append(corrupted)

            candidates = deepcopy(currant_candidate)

        if candidates:
            pred_probs = []
            for c in candidates:
                model_input, weight_of_embed_codes = self.input_handle(c)
                pred_prob = self.model(model_input, weight_of_embed_codes)
                pred_probs.append(pred_prob)

            target_probs = []
            for p in pred_probs:
                target_probs.append(p[0][1 - y])
            best_candidate_id = np.argmax(target_probs)
            log_pred_prob = target_probs[best_candidate_id]

            new_person = candidates[best_candidate_id]
        else:
            print("empty candidates!", file=log_f, flush=True)

        changed_pos = set()

        for v in range(len(new_person)):
            for code in new_person[v]:
                if code not in person[v]:
                    changed_pos.add((v, code))
        for v in range(len(person)):
            for code in person[v]:
                if code not in new_person[v]:
                    changed_pos.add((v, code))

        return new_person, log_pred_prob, changed_pos

    def attack(self, person, y):
        st = time.time()
        person_before = deepcopy(person)
        best_person = deepcopy(person)
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
            return person, pred_prob, 0, success_flag, 0, {}

        best_score = 1 - pred_prob

        changed_pos = set()

        iteration = 0

        recompute = True

        n_change = 0

        while (pred == y or best_score < TAU):

            iteration += 1

            if recompute:
                model_input, weight_of_embed_codes = self.input_handle(person)
                embed_codes = self.model.embed(torch.LongTensor(model_input))

                weight_of_embed_codes = Variable(weight_of_embed_codes.data, requires_grad=True)

                weighted_embed_codes = embed_codes * torch.unsqueeze(weight_of_embed_codes, dim=3)

                output = self.forward_lstm(weighted_embed_codes, self.model)

                loss = self.criterion(output, Variable(torch.LongTensor([y])))

                loss.backward()

                score = np.zeros((len(person), 4130))

            for visit_idx in range(len(one_hot_codes)):
                for code_idx in range(len(one_hot_codes[visit_idx])):
                    pos = (visit_idx, code_idx)
                    if pos in changed_pos:
                        score[visit_idx][code_idx] = 0
                        continue
                    a = weight_of_embed_codes.grad.data[visit_idx][0][code_idx]

                    score[visit_idx][code_idx] = a.data

            indices = np.argsort(-np.reshape(score, (-1)))

            person, pred_prob, temp_changed_pos = self.code_paraphrase(one_hot_codes, person, indices[:INDICE_K], y)

            if pred_prob > best_score:

                for pos in temp_changed_pos:
                    changed_pos.add(pos)

                best_score = pred_prob
                n_change += len(temp_changed_pos)
                best_person = deepcopy(person)
                recompute = True
            else:
                person = deepcopy(best_person)
                recompute = False

            if best_score > 0.5:
                pred = 1 - y

            if (time.time() - st) > SECONDS and (pred == y or pred_prob < TAU):
                print("===== Time out! Attack Fail =====", file=log_f, flush=True)
                success_flag = -1
                break

        # flag = 0
        # for v_idx in range(len(best_person)):
        #     for code in person_before[v_idx]:
        #         if code not in best_person[v_idx]:
        #             flag = 1
        #             break
        #     if flag:
        #         break
        # if flag:
        #     print("=== Some original codes is reduced (from 1 to 0). ===")

        return best_person, best_score, n_change, success_flag, iteration, changed_pos

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

    # # ****** #
    # file1 = './AttackCodes/codes_QL_sub_k=10_t=0.5_s=180.pickle'
    #
    # attack_code_dict = pickle.load(open(file1, 'rb'))
    #
    # file2 = './Each_TimeNumber/time_QL_sub_k=10_t=0.5_s=180.pickle'
    #
    # saving_time = pickle.load(open(file2, 'rb'))
    # # ****** #

    for i in range(n_people):
        print("-------- %d ---------" % (i), file=log_f, flush=True)

        person = test[0][i]

        y = test[1][i]

        n_visit = len(person)

        print('* Processing:%d/%d person, number of visit for this person: %d' % (i, n_people, n_visit), file=log_f, flush=True)

        print("* Original: "+str(person), file=log_f, flush=True)

        print("  Original label: %d" % (y), file=log_f, flush=True)

        st = time.time()
        changed_person, score, num_changed, success_flag, iteration, changed_pos = attacker.attack(person, y)

        et = time.time()
        all_t = et-st

        if success_flag == 1:
            n_success += 1
            n_iteration += iteration
            total_node_change += num_changed

            saving_time[i] = all_t
            attack_code_dict[i] = list(changed_pos)
        elif success_flag == -1:
            n_fail += 1

        print("* Result: ", file=log_f, flush=True)

        print(changed_pos, file=log_f, flush=True)
        print(changed_person, file=log_f, flush=True)

        print(score, file=log_f, flush=True)

        print("  Nnumber of changed codes: %d" % (num_changed), file=log_f, flush=True)

        print("  Number of iterations for this: " + str(iteration), file=log_f, flush=True)

        print(" Time: "+str(all_t), file=log_f, flush=True)

        print("* SUCCESS Number NOW: %d " % (n_success), file=log_f, flush=True)
        print("* Failure Number NOW: %d " % (n_fail), file=log_f, flush=True)

        if n_success:
            print("  Average Number of success changed codes: " + str(float(total_node_change)/float(n_success)), file=log_f, flush=True)
            print("  Average Number of success iterations: " + str(float(n_iteration) / float(n_success)), file=log_f, flush=True)

        if i % 20 == 0:
            pickle.dump(attack_code_dict, open('./AttackCodes/codes_QL_%s_k=%d_t=%s_s=%d.pickle' % (MODEL_TYPE, INDICE_K, str(TAU), SECONDS), 'wb'))
            pickle.dump(saving_time, open('./Each_TimeNumber/time_QL_%s_k=%d_t=%s_s=%d.pickle' % (MODEL_TYPE, INDICE_K, str(TAU), SECONDS), 'wb'))

    pickle.dump(attack_code_dict, open('./AttackCodes/codes_QL_%s_k=%d_t=%s_s=%d.pickle' % (MODEL_TYPE, INDICE_K, str(TAU), SECONDS), 'wb'))
    pickle.dump(saving_time, open('./Each_TimeNumber/time_QL_%s_k=%d_t=%s_s=%d.pickle' % (MODEL_TYPE, INDICE_K, str(TAU), SECONDS), 'wb'))

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