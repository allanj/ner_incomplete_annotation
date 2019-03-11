import dynet as dy

class CharRNN:

    def __init__(self, config, model):
        self.char_emb_size = config.char_emb_size
        self.char2idx = config.char2idx
        self.chars = config.idx2char
        self.char_size = len(self.chars)
        self.model = model
        self.char_emb = model.add_lookup_parameters((self.char_size, self.char_emb_size))
        self.bilstm = dy.BiRNNBuilder(1, self.char_emb_size, config.charlstm_hidden_dim, self.model, dy.LSTMBuilder)

        self.fw_lstm = dy.CompactVanillaLSTMBuilder(1, self.char_emb_size, config.charlstm_hidden_dim/2, self.model)
        self.bw_lstm = dy.CompactVanillaLSTMBuilder(1, self.char_emb_size, config.charlstm_hidden_dim/2, self.model)

        print("char embedding size: %d" % (self.char_emb_size))
        print("char hidden size: %d" % (config.charlstm_hidden_dim))

    def forward_char(self, x):
        embeddings = [self.char_emb[c] for c in x]
        fw_state = self.fw_lstm.initial_state()
        bw_state = self.bw_lstm.initial_state()
        fw_out = fw_state.transduce(embeddings)
        # bw_out = bw_state.transduce(list(reversed(embeddings)))
        bw_out = bw_state.transduce(embeddings[::-1])


        return fw_out[len(x) - 1], bw_out[len(x) - 1]

