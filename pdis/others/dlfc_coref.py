from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_parameters
from colab_eval_metrics import EPSILON

class CharCNN(nn.Module):
    """Character-level CNN. Contains character embeddings.
    """

    def __init__(self, filter_size, filter_widths, char_dict, char_embedding_size, max_word_length):
        super().__init__()

        # I assumed character dict already includes padding 0 as padding, I know it includes unknown
        self.embeddings = nn.Embedding(len(char_dict)+1, char_embedding_size, padding_idx=0)

        # TODO; figure out a way to have varying lenght on in-channels, I'm sure it's possible
        # for now I padded the words to a maximum length
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=max_word_length,
                                              out_channels=filter_size,
                                              kernel_size=n) for n in filter_widths])

    def forward(self, sent_as_char_index):
        """ Compute filter-dimensional character-level features for each doc token """
        char_emb = self.embeddings(sent_as_char_index)
        num_sents = char_emb.shape[0]
        max_sent_length = char_emb.shape[1]
        char_emb = char_emb.reshape(num_sents*max_sent_length, char_emb.shape[2], char_emb.shape[3])
        convolved = torch.cat([F.relu(conv(char_emb)) for conv in self.convs], dim=2)
        # "they" say no need to unpad as max pooling handles it. I really have no idea, need to read up on CNNs!
        pooled = F.max_pool1d(convolved, convolved.shape[2])
        return pooled.reshape(num_sents, max_sent_length, pooled.shape[-2])



class DocumentEncoder(nn.Module):
    """This model is was designed to be run over a list of sentences,
    i.e. a document, each sentence is represented by a list of embeddings.
    """
    def __init__(self, lstm_size, lstm_dropout, emb_size):
        super().__init__()

        output_size = lstm_size*2
        self.highway_projection = nn.Sequential(nn.Linear(output_size, output_size), nn.Sigmoid())

        self.layer_0_bilstm = nn.LSTM(emb_size, lstm_size, bidirectional=True, batch_first=True)
        self.layer_1_bilstm = nn.LSTM(output_size, lstm_size, bidirectional=True, batch_first=True)
        self.layer_2_bilstm = nn.LSTM(output_size, lstm_size, bidirectional=True, batch_first=True)

        self.lstm_dropout = lstm_dropout
        self.lstms = [self.layer_0_bilstm, self.layer_1_bilstm, self.layer_2_bilstm]

    def forward(self, text_emb, text_lens, is_training):
        current_inputs = text_emb
        for layer in range(len(self.lstms)):
            text_outputs, _ = self.lstms[layer](current_inputs)
            if is_training:
                text_outputs = nn.Dropout(self.lstm_dropout)(text_outputs)
            if layer > 0:
                # reshaping document into [num_words, emb_size]
                batch_size = text_outputs.shape[0]
                seqlen = text_outputs.shape[1]
                emb_size = text_outputs.shape[2]
                reshaped = torch.reshape(text_outputs, [batch_size*seqlen, emb_size])
                highway_gates = self.highway_projection(reshaped).reshape(batch_size, seqlen, emb_size)
                text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs

            current_inputs = text_outputs
        return flatten_emb_by_sentence(text_outputs, text_lens)



class SpanRepresentation(nn.Module):
    """A learned representation for the mention. A mention is represented by
       torch.cat([states_start_end, attention_embed, mention_distance])
       This assumes that the candidate mentions are all less than max_width
       in length.
    """
    def __init__(self, max_span_width, feature_size, context_emb_size, dropout):
        super().__init__()

        self.dropout = dropout
        self.max_span_width = max_span_width
        self.span_width_embeddings = nn.Embedding(max_span_width, feature_size)
        self.attention = nn.Linear(context_emb_size, 1)

    def forward(self, head_emb, text_lens, context_outputs, span_starts, span_ends, is_training):
        """ I use all features and model heads
        """
        span_width = 1 + span_ends - span_starts #[k]
        span_width_index = span_width - 1 # -1 just so it starts from 0
        max_tensor = to_cuda(torch.zeros_like(span_width) + self.max_span_width - 1)
        span_width_index = torch.where(span_width_index < max_tensor, span_width_index, max_tensor)
        span_width_emb = self.span_width_embeddings(span_width_index)
        if is_training:
            span_width_emb = nn.Dropout(self.dropout)(span_width_emb)

        # for each span, span indices are the indices of all the possible words in the span
        span_indices = to_cuda(torch.arange(self.max_span_width).unsqueeze(0)) + span_starts.unsqueeze(1)
        span_indices = torch.min(torch.zeros_like(span_indices)+context_outputs.shape[0]-1, span_indices)
        span_text_emb = head_emb[span_indices]
        head_scores = self.attention(context_outputs)  #[num_words, 1]

        span_head_scores = head_scores[span_indices]   #[k, max_span_width, 1]
        span_mask = mask_len(span_width, self.max_span_width).float()
        span_mask = span_mask.unsqueeze(1).transpose(2,1)

        span_head_scores += span_mask.log()
        span_attention = F.softmax(span_head_scores, 1)  # [k, max_span_width, 1]
        span_head_emb = torch.sum((span_attention*span_text_emb), 1)
        representation = torch.cat((context_outputs[span_starts], context_outputs[span_ends], span_width_emb, span_head_emb), dim=1) # [k, emb]
        return representation


class DenseLayer(nn.Module):
    def __init__(self, embeds_dim, hidden_size, hidden_dropout, num_outputs=1):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(embeds_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size, num_outputs)
        )

    def forward(self, x):
        return self.score(x)



class ISModel(nn.Module):
    def __init__(self, exp='stl', smythe=None):
        super().__init__()

        self.hidden_layer_size = 150
        self.dropout = 0.5
        self.lexical_dropout = 0.5
        self.lstm_dropout = 0.2
        self.char_embedding_size = 8
        self.feat_emb_size = 27 # the one hot features
        self.filter_size = 50
        self.filter_widths = model_parameters.filter_widths
        self.lm_size = 3
        self.genres = model_parameters.genres
        self.feature_size = 20
        self.max_span_width = 30
        self.contextualization_size = 200
        self.glove_dim = 300
        self.lm_size = 1024
        self.lm_layers = 3
        self.num_classes = model_parameters.num_classes
        self.num_users = model_parameters.num_users

        # other paramters
        self.char_dict = load_char_dict(model_parameters.char_vocab_path)

        self.lm_weights = nn.Parameter(torch.zeros(self.lm_layers))
        self.lm_scaling = nn.Parameter(torch.tensor(1.0)) # a zero dimension tensor

        #self.genre_embeddings = nn.Embedding(len(self.genres), self.feature_size)

        # mini models
        if self.char_embedding_size > 0:
            lstm_emb_size = self.lm_size + self.glove_dim + self.filter_size
            head_emb_size = self.glove_dim + self.filter_size
        else:
            lstm_emb_size = self.lm_size + self.glove_dim
            head_emb_size = self.glove_dim
        span_emb_dim = self.contextualization_size*4 + head_emb_size + self.feature_size + self.feat_emb_size

        self.char_cnn = CharCNN(self.filter_size, self.filter_widths, self.char_dict, self.char_embedding_size, model_parameters.max_word_length)
        self.lstm_contextualize = DocumentEncoder(self.contextualization_size, self.lstm_dropout, lstm_emb_size)
        context_emb_size = self.contextualization_size*2
        self.mention_encoder = SpanRepresentation(self.max_span_width, self.feature_size, context_emb_size, self.dropout)

        self.scorer = DenseLayer(span_emb_dim, self.hidden_layer_size, self.dropout, self.num_classes)

        # For the crowd layer
        beta = torch.eye(self.num_classes)
        self.global_user_confusion = nn.Parameter(beta.unsqueeze(0).repeat([self.num_users, 1, 1]))

    def forward(self, context_word_emb, head_word_emb, lm_emb, char_index, text_lens, gold_starts, gold_ends, features, anno_users, anno_status, is_training):
        num_sentences = context_word_emb.shape[0]
        max_sentence_length = context_word_emb.shape[1]

        if self.char_embedding_size > 0: # if using character embeddings
            aggregated_char_emb = self.char_cnn(char_index)  #[num_sentences, max_sentence_length, emb]

        lm_emb_size = lm_emb.shape[2]
        lm_num_layers = lm_emb.shape[3]
        # aggregating, flattening and scaling the lm_embeddings
        lm_emb = lm_emb.reshape([num_sentences*max_sentence_length*lm_emb_size, lm_num_layers])
        lm_emb = torch.matmul(lm_emb, self.lm_weights.unsqueeze(1)) # [num_sentences * max_sentence_length * emb, 1]
        lm_emb = lm_emb.reshape([num_sentences, max_sentence_length, lm_emb_size])
        lm_emb *= self.lm_scaling

        if self.char_embedding_size > 0: # could have done it with the if statement above but memory things!
           context_emb = torch.cat([context_word_emb,aggregated_char_emb,lm_emb], 2) # [num_sentences, max_sentence_length, emb]
           head_emb = torch.cat([head_word_emb,aggregated_char_emb],2) # [num_sentences, max_sentence_length, emb]
        else:
           context_emb = torch.cat([context_word_emb,lm_emb], 2) # [num_sentences, max_sentence_length, emb]
           head_emb = torch.cat([head_word],2) # [num_sentences, max_sentence_length, emb]

        if is_training:
            context_emb = nn.Dropout(self.lexical_dropout)(context_emb)
            head_emb = nn.Dropout(self.lexical_dropout)(head_emb)

        context_outputs = self.lstm_contextualize(context_emb, text_lens, is_training) # [num_words, emb]

        span_emb = self.mention_encoder(flatten_emb_by_sentence(head_emb, text_lens), text_lens, context_outputs, gold_starts, gold_ends, is_training)
        span_emb = torch.cat([span_emb, features], 1)

        information_status_scores =  self.scorer(span_emb) # [k, 4]
        softmax_scores = F.softmax(information_status_scores, 1)  +  EPSILON

        if is_training:
            # THE CROWD LAYER
            num_mentions = anno_users.shape[0]
            max_anno_length = anno_users.shape[1]

            # flatten the user and class information
            flattened_anno_users = torch.flatten(anno_users) # [num_mentions, max_anno_length]
            flattened_anno_status = torch.flatten(anno_status)  # [num_mentions, max_anno_length]

            # use the mask to remove the useless padding-injected information from the user and the class tensors; one hot encode the latter
            valid_users = flattened_anno_users[flattened_anno_users>-1]         # [across_mention_annotations]
            valid_judgements = flattened_anno_status[flattened_anno_status>-1]     # [across_mention_annotations]
            judgements_one_hot = to_cuda(torch.eye(self.num_classes)[valid_judgements])     # [across_mention_annotations, 4]

            # extract the confusion matrices for the users who provided the annotations
            users_confusions = torch.index_select(self.global_user_confusion, 0, valid_users)  # [across_mention_annotations, 4, 4]

            # the scores produced by the neural network for each mention are used in the computation of every expected annotator response; so duplicate them appropriately
            duplicated_mention_scores = information_status_scores.unsqueeze(1)   # [num_mentions, 1, 4]
            duplicated_mention_scores = duplicated_mention_scores.repeat(1, max_anno_length, 1)    # [num_mentions, max_anno_length, 4]
            duplicated_mention_scores = torch.reshape(duplicated_mention_scores, (num_mentions*max_anno_length, self.num_classes))  # [num_mentions * max_anno_length, 4]

            # corresponding mention scores for each annotation provided
            selected_scores = duplicated_mention_scores[flattened_anno_status>-1]    # [across_mention_annotations, 4]
            # the expected annotator respone is a linear combination of the annotator confusion and the neural network prediction
            expected_user_responses = torch.matmul(users_confusions, selected_scores.unsqueeze(2)).squeeze(2)  # [across_mention_annotations, 4]
            expected_user_responses = F.softmax(expected_user_responses, 1) + EPSILON

            cross_entropy = torch.mul(judgements_one_hot, expected_user_responses.log())
            loss  = -torch.sum(cross_entropy)


            return softmax_scores, None, loss, None
        else:
            return softmax_scores, None, None, None

