from typing import Dict, Optional, List, Any

import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, InputVariationalDropout
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum, replace_masked_values
from allennlp.training.metrics import CategoricalAccuracy

import torch.nn as nn
from torch.autograd import Variable

class MyActivationFunction(nn.Module):

    def __init__(self):
        super(MyActivationFunction, self).__init__()

    def forward(self, x):
        #gauss = torch.exp((-(x - self.mean) ** 2)/(2* self.std ** 2))
        #return torch.clamp(gauss, min=self.min, max=self.max)

        m = nn.LeakyReLU(0.1)
        return 1-m(1-m(x))

@Model.register("esimModified")
class ESIMModified(Model):
    """
    This ``Model`` implements the ESIM sequence model described in `"Enhanced LSTM for Natural Language Inference"
    <https://www.semanticscholar.org/paper/Enhanced-LSTM-for-Natural-Language-Inference-Chen-Zhu/83e7654d545fbbaaf2328df365a781fb67b841b4>`_
    by Chen et al., 2017.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    encoder : ``Seq2SeqEncoder``
        Used to encode the premise and hypothesis.
    similarity_function : ``SimilarityFunction``
        This is the similarity function used when computing the similarity matrix between encoded
        words in the premise and words in the hypothesis.
    projection_feedforward : ``FeedForward``
        The feedforward network used to project down the encoded and enhanced premise and hypothesis.
    inference_encoder : ``Seq2SeqEncoder``
        Used to encode the projected premise and hypothesis for prediction.
    output_feedforward : ``FeedForward``
        Used to prepare the concatenated premise and hypothesis for prediction.
    output_logit : ``FeedForward``
        This feedforward network computes the output logits.
    dropout : ``float``, optional (default=0.5)
        Dropout percentage to use.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 similarity_function: SimilarityFunction,
                 projection_feedforward: FeedForward,
                 inference_encoder: Seq2SeqEncoder,
                 output_feedforward: FeedForward,
                 output_logit: FeedForward,
                 dropout: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                    similarity_weight : int = 30) -> None:
        super().__init__(vocab, regularizer)
        
        self.label_map = vocab.get_token_to_index_vocabulary('labels')

        label_map = [None]*len(self.label_map)
        for lb,lb_idx in self.label_map.items():
            label_map[lb_idx] = lb
        self.label_map = label_map

        self._text_field_embedder = text_field_embedder
        self._encoder = encoder
        print(similarity_function) 
        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        self._projection_feedforward = projection_feedforward

        self._inference_encoder = inference_encoder

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
            self.rnn_input_dropout = InputVariationalDropout(dropout)
        else:
            self.dropout = None
            self.rnn_input_dropout = None

        self._output_feedforward = output_feedforward
        self._output_logit = output_logit

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        check_dimensions_match(encoder.get_output_dim() * 4, projection_feedforward.get_input_dim(),
                               "encoder output dim", "projection feedforward input")
        check_dimensions_match(projection_feedforward.get_output_dim(), inference_encoder.get_input_dim(),
                               "proj feedforward output dim", "inference lstm input dim")

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

        self.lambda_layer = nn.Sequential(nn.Linear(16, 1,bias=False), MyActivationFunction())
        
        self.lambda_layer[0].weight.data = torch.tensor([[0.1,0.5,0.5,0.5, 0.5,0.1,0.5,0.5, 0.5,0.5,0.1,0.5, 0.5,0.5,0.5,0.9]])
        self.similarity_weight = similarity_weight
        print("SIMILARITY WEIGHT BEING USED IS : {0}".format(self.similarity_weight)) 
    def createNERFeatVecAndSimMat(self, p, h, metadata, simWeight):
        numOfSamples, maxLenP = 0,0
        if 'tokens' in p:
            numOfSamples, maxLenP = p['tokens'].shape
            numOfSamples, maxLenH = h['tokens'].shape
        else:
            numOfSamples, maxLenP, _ = p['elmo'].shape
            numOfSamples, maxLenH, _ = h['elmo'].shape
        #numOfSamples, maxLenP, = p['tokens'].shape if 'tokens' in p else p['elmo'].shape
        #numOfSamples, maxLenH, = h['tokens'].shape if 'tokens' in h else h['elmo'].shape
        #print("Number of Samples : {0}".format(numOfSamples))
        #print("Max length for Premise : {0}".format(maxLenP))
        #print("Max length for Hypothesis : {0}".format(maxLenH))
        for i in range(numOfSamples):
            lsent_unaryWordFeatures = metadata[i]["premiseUF"]
            rsent_unaryWordFeatures = metadata[i]["hypothesisUF"]
            #print("Premise UF : {0}".format(lsent_unaryWordFeatures))
            #print("Hypothesis UF : {0}".format(rsent_unaryWordFeatures))
            lsent = metadata[i]["premise_tokens"]
            rsent = metadata[i]["hypothesis_tokens"]
            
            #print("Premise Tokens : {0}".format(lsent))
            #print("Hypothesis Tokens : {0}".format(rsent))

            lsent = lsent + ['oov' for k in range(maxLenP - len(lsent))]
            rsent = rsent + ['oov' for k in range(maxLenH - len(rsent))]

            if len(lsent_unaryWordFeatures) <= maxLenP:
                lsent_unaryWordFeatures = lsent_unaryWordFeatures + [[0,0,0,1] for k in range(maxLenP - len(lsent_unaryWordFeatures))]
            elif len(lsent_unaryWordFeatures) > maxLenP:
                lsent_unaryWordFeatures = lsent_unaryWordFeatures[:maxLenP]

            if len(rsent_unaryWordFeatures) <= maxLenH:
                rsent_unaryWordFeatures = rsent_unaryWordFeatures + [[0,0,0,1] for k in range(maxLenH - len(rsent_unaryWordFeatures))]
            elif len(rsent_unaryWordFeatures) > maxLenH:
                rsent_unaryWordFeatures = rsent_unaryWordFeatures[:maxLenH]

            #print("Premise UF After Padding : {0}".format(lsent_unaryWordFeatures))
            #print("Hypothesis UF After Padding : {0}".format(rsent_unaryWordFeatures))            

            if i==0:
                ner_feature_vector = self.compute_feature_vector_mod16Dim(lsent_unaryWordFeatures, rsent_unaryWordFeatures)
                ner_feature_vectors = torch.unsqueeze(ner_feature_vector,0)

                sim=self.symbolic_similarity_matrix(lsent,rsent,simWeight)
                sim_matrices = torch.unsqueeze(sim,0)
            else:
                ner_feature_vector = self.compute_feature_vector_mod16Dim(lsent_unaryWordFeatures, rsent_unaryWordFeatures)
                ner_feature_vector = torch.unsqueeze(ner_feature_vector,0)
                ner_feature_vectors = torch.cat((ner_feature_vectors, ner_feature_vector))

                sim=self.symbolic_similarity_matrix(lsent,rsent,simWeight)
                sim_matrice = torch.unsqueeze(sim,0)
                sim_matrices = torch.cat((sim_matrices, sim_matrice))
        sim_matrices=Variable(sim_matrices)
        ner_feature_vectors=Variable(ner_feature_vectors)
        if torch.cuda.is_available():
            sim_matrices=sim_matrices.cuda()
            ner_feature_vectors=ner_feature_vectors.cuda()
        return ner_feature_vectors,sim_matrices

    def symbolic_similarity_matrix(self, sent1: list, sent2: list, simWeight=1):
        """
        Computes symbolic similarity matrix for sentence 1 and 2
        :param sent1: list of tokens in sentence 1
        :param sent2: list of tokens in sentence 2
        :return: numpy array of size (len(sent1), len(sent2)) whose elements are either 1 or 0 for symbolic similarity
        """
        sim_matrix = torch.zeros([len(sent1), len(sent2)], dtype=torch.uint8)
        for i in range(len(sent1)):
            for j in range(len(sent2)):
                if sent1[i] == sent2[j]:
                    sim_matrix[i][j] = simWeight
        return sim_matrix

    """
    This function produces a 16 Dim feature vector.
    """
    def compute_feature_vector_mod16Dim(self, tags1, tags2):
        feature_vec_per_sent = torch.zeros(0)
        for tag1 in tags1:
            idx1 = tag1.index(1)
            for tag2 in tags2:
                idx2 = tag2.index(1)
                temp = [0]*16
                temp[(idx1*4)+idx2] = 1

                if feature_vec_per_sent.shape[0] == 0:
                    feature_vec_per_sent = torch.tensor(temp).view(1,16)
                else:
                    feature_vec_per_sent = torch.cat((feature_vec_per_sent, torch.tensor(temp).view(1, 16)), 0)
        return feature_vec_per_sent


    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None  # pylint:disable=unused-argument
               ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            From a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata containing the original tokenization of the premise and
            hypothesis with 'premise_tokens' and 'hypothesis_tokens' keys respectively.

        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_premise = self._text_field_embedder(premise)
        embedded_hypothesis = self._text_field_embedder(hypothesis)
        premise_mask = get_text_field_mask(premise).float()
        hypothesis_mask = get_text_field_mask(hypothesis).float()
        
        #print("\n")
        #print("Raw Premise")
        #print(premise)    
        #print("\n")
        #print("Raw Hypothesis")
        #print(hypothesis)
        #print("\n")
        
        #print("\n")
        #print("Raw Premise length")
        #print(len(premise))
        #print("\n")
        #print("Raw Hypothesis length")
        #print(len(hypothesis))
        #print("\n")
        
        #print("metadata")
        #print(metadata)
        
        NER_Features,symbolic_similarity_mat = self.createNERFeatVecAndSimMat(premise,hypothesis,metadata,self.similarity_weight)

        #print(NER_Features.shape)
        #print(symbolic_similarity_mat.shape)
        
        float_NER = NER_Features.float()
        
        lambda_probabilities = self.lambda_layer(float_NER)
        lambda_probabilities = (torch.transpose(lambda_probabilities,1,2)).reshape((symbolic_similarity_mat.shape[0],
                                                                                    symbolic_similarity_mat.shape[1],
                                                                                    symbolic_similarity_mat.shape[2]))
        
        #print("Lambda Prob")
        #print(lambda_probabilities.shape)
        
        # apply dropout for LSTM
        if self.rnn_input_dropout:
            embedded_premise = self.rnn_input_dropout(embedded_premise)
            embedded_hypothesis = self.rnn_input_dropout(embedded_hypothesis)

        # encode premise and hypothesis
        encoded_premise = self._encoder(embedded_premise, premise_mask)
        encoded_hypothesis = self._encoder(embedded_hypothesis, hypothesis_mask)

        # Shape: (batch_size, premise_length, hypothesis_length)
        similarity_matrix = self._matrix_attention(encoded_premise, encoded_hypothesis)
        #print(similarity_matrix.shape)
        float_symbolic = symbolic_similarity_mat.float()
        similarity_matrix1 = similarity_matrix
        similarity_matrix = torch.add(torch.mul(lambda_probabilities, similarity_matrix), torch.mul(-1 * torch.sub(lambda_probabilities, 1),float_symbolic))

        #print(similarity_matrix.shape)        

        # Shape: (batch_size, premise_length, hypothesis_length)
        p2h_attention = masked_softmax(similarity_matrix, hypothesis_mask)
        # Shape: (batch_size, premise_length, embedding_dim)
        attended_hypothesis = weighted_sum(encoded_hypothesis, p2h_attention)

        # Shape: (batch_size, hypothesis_length, premise_length)
        h2p_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
        # Shape: (batch_size, hypothesis_length, embedding_dim)
        attended_premise = weighted_sum(encoded_premise, h2p_attention)

        # the "enhancement" layer
        premise_enhanced = torch.cat(
                [encoded_premise, attended_hypothesis,
                 encoded_premise - attended_hypothesis,
                 encoded_premise * attended_hypothesis],
                dim=-1
        )
        hypothesis_enhanced = torch.cat(
                [encoded_hypothesis, attended_premise,
                 encoded_hypothesis - attended_premise,
                 encoded_hypothesis * attended_premise],
                dim=-1
        )

        # The projection layer down to the model dimension.  Dropout is not applied before
        # projection.
        projected_enhanced_premise = self._projection_feedforward(premise_enhanced)
        projected_enhanced_hypothesis = self._projection_feedforward(hypothesis_enhanced)

        # Run the inference layer
        if self.rnn_input_dropout:
            projected_enhanced_premise = self.rnn_input_dropout(projected_enhanced_premise)
            projected_enhanced_hypothesis = self.rnn_input_dropout(projected_enhanced_hypothesis)
        v_ai = self._inference_encoder(projected_enhanced_premise, premise_mask)
        v_bi = self._inference_encoder(projected_enhanced_hypothesis, hypothesis_mask)

        # The pooling layer -- max and avg pooling.
        # (batch_size, model_dim)
        v_a_max, _ = replace_masked_values(
                v_ai, premise_mask.unsqueeze(-1), -1e7
        ).max(dim=1)
        v_b_max, _ = replace_masked_values(
                v_bi, hypothesis_mask.unsqueeze(-1), -1e7
        ).max(dim=1)

        v_a_avg = torch.sum(v_ai * premise_mask.unsqueeze(-1), dim=1) / torch.sum(
                premise_mask, 1, keepdim=True
        )
        v_b_avg = torch.sum(v_bi * hypothesis_mask.unsqueeze(-1), dim=1) / torch.sum(
                hypothesis_mask, 1, keepdim=True
        )

        # Now concat
        # (batch_size, model_dim * 2 * 4)
        v_all = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        # the final MLP -- apply dropout to input, and MLP applies to output & hidden
        if self.dropout:
            v_all = self.dropout(v_all)

        output_hidden = self._output_feedforward(v_all)
        label_logits = self._output_logit(output_hidden)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)
    
        label_map = [self.label_map] * len(label_probs)
        lambda_layer_learnt_weights = [self.lambda_layer[0].weight.data]*len(label_probs)
       
        output_dict = {"label_logits": label_logits,
                        "label_probs": label_probs,
                        "label_names":label_map,
                        "lambda_layer_learnt_weights":lambda_layer_learnt_weights,
                        "eij":similarity_matrix1,
                        "finalEij":similarity_matrix,
                        "LambdaProb":lambda_probabilities}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
