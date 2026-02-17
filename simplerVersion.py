import sys
import numpy as np



# this reads the fasta file

def read_fasta(filepath):
    sequences = []
    current = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current).upper())
                    current = []
            else:
                current.append(line)

        if current:
            sequences.append("".join(current).upper())

    return sequences



# encode DNA sequences into integer arrays (A=0, C=1, G=2 and T=3)

def encode_sequences(sequences):
    mapping = np.full(256, -1, dtype=np.int8)
    mapping[ord("A")] = 0
    mapping[ord("C")] = 1
    mapping[ord("G")] = 2
    mapping[ord("T")] = 3

    seq_array = np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8)
    encoded = mapping[seq_array]

    lengths = np.array([len(s) for s in sequences])
    seq_length = lengths[0]

    encoded = encoded.reshape(len(sequences), seq_length)
    return encoded



# train markov model of order m

def train_markov_model(encoded, m):
    n_sequences, seq_length = encoded.shape

    if m == 0:
        counts = np.ones((1, 4))
        bases = encoded.ravel()
        counts[0] += np.bincount(bases, minlength=4)

    else:
        n_contexts = 4 ** m
        counts = np.ones((n_contexts, 4))

        powers = 4 ** np.arange(m - 1, -1, -1)

        contexts = np.lib.stride_tricks.sliding_window_view(
            encoded, window_shape=m, axis=1
        )

        contexts = contexts[:, :-1, :]
        context_indices = np.tensordot(contexts, powers, axes=([2], [0]))

        next_bases = encoded[:, m:]

        np.add.at(
            counts,
            (context_indices.ravel(), next_bases.ravel()),
            1
        )

    probs = counts / counts.sum(axis=1, keepdims=True)
    return np.log(probs)



# compute log-likelihood of sequences under the model

def compute_log_likelihood(encoded, log_probs, m):

    if m == 0:
        return np.sum(log_probs[0][encoded], axis=1)

    powers = 4 ** np.arange(m - 1, -1, -1)

    contexts = np.lib.stride_tricks.sliding_window_view(
        encoded, window_shape=m, axis=1
    )

    contexts = contexts[:, :-1, :]
    context_indices = np.tensordot(contexts, powers, axes=([2], [0]))

    next_bases = encoded[:, m:]

    scores = log_probs[context_indices, next_bases]
    return scores.sum(axis=1)



# main function to read sequences, train the model and compute scores
 
def main():
    if len(sys.argv) != 3:
        print("Usage: python simplerVersion.py <fasta_file> <order_m>")
        sys.exit(1)

    fasta_file = sys.argv[1]
    m = int(sys.argv[2])

    sequences = read_fasta(fasta_file)
    encoded = encode_sequences(sequences)

    log_probs = train_markov_model(encoded, m)
    scores = compute_log_likelihood(encoded, log_probs, m)

    for s in scores:
        print(s)


if __name__ == "__main__":
    main()
