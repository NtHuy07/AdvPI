import random
import numpy as np
from nltk.tokenize import RegexpTokenizer

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Error(Exception):
    """Base class for other exceptions"""
    pass

class WordNotInDictionaryException(Error):
    """Raised when the input value is too small"""
    pass


embedder = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# ---------- get embeddings for all vocab tokens ----------
def build_vocab_embeddings():
    vocab = list(tokenizer.get_vocab().keys())

    embeddings = embedder.encode(vocab, batch_size=512, convert_to_numpy=True)
    return vocab, embeddings


vocab, vocab_embeddings = build_vocab_embeddings()


class AttackUtils():
    def __init__(
        self,
        orig_prompt
    ):

        tokenizer = RegexpTokenizer(r'\w+')
        self.orig_tokens = tokenizer.tokenize(orig_prompt.lower())


    def replaceWithBug(self, x_prime, word_idx, bug):
        return x_prime[:word_idx] + [bug] + x_prime[word_idx + 1:]


    def selectBug(self, word, word_pos, mode="full"):
        
        if mode == "full":
            target_num = random.randint(0, 5)
        elif mode == "char-only":
            target_num = random.randint(0, 4)
        elif mode == "token-only":
            target_num = 5
        else:
            raise ValueError(f"No variation mode named {mode}")

        if target_num != 5: # Character-level mutation
            word_slice = word.split("_")
            #randomly choose one
            subword_idx = np.random.choice(len(word_slice), size=1)[0]
            subword = word_slice[subword_idx]

            if target_num == 0:  
                bug = self.bug_convert_to_leet(subword)
            elif target_num == 1:
                bug = self.bug_insert(subword)
            elif target_num == 2:
                bug = self.bug_delete(subword)
            elif target_num == 3:
                bug = self.bug_sub_C(subword)
            elif target_num == 4:
                bug = self.bug_swap(subword)

            word_slice[subword_idx] = bug
            bugs = '_'.join(word_slice)

        else:   

            bugs = self.nearest_substitutes(self.orig_tokens[word_pos])

        return bugs

    def nearest_substitutes(self, token, top_k=10, sim_threshold=0.6):
        # embed original token
        token_vec = embedder.encode([token], show_progress_bar=False)

        # cosine similarity to all vocab tokens
        sims = cosine_similarity(token_vec, vocab_embeddings)[0]

        # sort indices by similarity
        idx_sorted = np.argsort(-sims)

        results = []
        sim_tokens = []
        for idx in idx_sorted:
            cand = vocab[idx]
            score = sims[idx]

            # skip identical token
            if cand == token:
                continue

            # optional filtering heuristics for realism
            if cand.startswith("##"):   # wordpiece fragments
                continue
            if score < sim_threshold:
                break

            results.append((cand, float(score)))
            sim_tokens.append(cand)

            if len(results) >= top_k:
                break
        
        if not sim_tokens:
            return token
        
        res = random.choice(sim_tokens)

        return res

    def bug_insert(self, word):
        res = word
        if len(word) <= 1:
            point = 0
        else:
            point = random.randint(0, len(word) - 1)
        if point == 0:
            res = "_" + res
        else:
            # insert _ instead " "
            res = res[0:point] + "_" + res[point:]
        return res

    def bug_delete(self, word):
        if len(word) <= 1:
            return ""
        res = word
        point = random.randint(0, len(word) - 1)
        if point == 0:
            res = res[1:]
        elif point == len(word) - 1:
            res = res[:-1]
        else:
            res = res[0:point] + res[point + 1:]
        return res

    def bug_swap(self, word):
        # If the string is too short, just return it
        if len(word) < 2:
            return word

        # Choose a random index i such that we can swap s[i] and s[i+1]
        i = random.randint(0, len(word) - 2)

        # Perform the swap
        swapped = list(word)
        swapped[i], swapped[i+1] = swapped[i+1], swapped[i]

        res = "".join(swapped)
        return res

    def bug_random_sub(self, word):
        res = word

        if len(word) <= 1:
            point = 0
        else:
            point = random.randint(0, len(word) - 1)

        choices = "qwertyuiopasdfghjklzxcvbnm"
        
        subbed_choice = choices[random.randint(0, len(list(choices)) - 1)]
        res = list(res)
        res[point] = subbed_choice
        res = ''.join(res)
        return res

    def bug_convert_to_leet(self, word):
        # Dictionary that maps each letter to its leet speak equivalent.
        
        if len(word) < 1:
            return word
        
        leet_dict = {
            'a': ['@'],
            'b': ['8'],
            'c': ['('],
            'd': ['c1'],
            'e': ['3'],
            'g': ['9'],
            'h': ['n'],
            'i': ['1'],
            'j': ['i'],
            'l': ['1'],
            'm': ['rn', 'nn'],
            'o': ['0'],
            's': ['5'],
            't': ['f'],
            'z': ['2'],
        }
        
        leet_idx = []
        for i, c in enumerate(word):
            if c in leet_dict.keys():
                leet_idx.append(i)
        if len(leet_idx) < 1:
            return word
        rnd_idx = np.random.choice(leet_idx)

        subbed = random.choice(leet_dict[word[rnd_idx].lower()])

        res = word[:rnd_idx] + subbed + word[rnd_idx + 1:]
        
        return res


    def bug_sub_C(self, word):

        if len(word) < 1:
            return word
        
        res = word

        if len(word) <= 1:
            point = 0
        else:
            point = random.randint(0, len(word) - 1)

        key_neighbors = self.get_key_neighbors()

        if word[point] not in key_neighbors:
            return word
        choices = key_neighbors[word[point]]
        subbed_choice = choices[random.randint(0, len(choices) - 1)]
        res = list(res)
        res[point] = subbed_choice
        res = ''.join(res)

        return res

    def get_key_neighbors(self):
        # By keyboard proximity
        neighbors = {
            "q": "was", "w": "qeasd", "e": "wrsdf", "r": "etdfg", "t": "ryfgh", "y": "tughj", "u": "yihjk",
            "i": "uojkl", "o": "ipkl", "p": "ol",
            "a": "qwszx", "s": "qweadzx", "d": "wersfxc", "f": "ertdgcv", "g": "rtyfhvb", "h": "tyugjbn",
            "j": "yuihknm", "k": "uiojlm", "l": "opk",
            "z": "asx", "x": "sdzc", "c": "dfxv", "v": "fgcb", "b": "ghvn", "n": "hjbm", "m": "jkn"
        }

        return neighbors