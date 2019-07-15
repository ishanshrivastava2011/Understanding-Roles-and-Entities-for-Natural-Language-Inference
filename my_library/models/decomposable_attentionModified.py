from typing import Dict, Optional, List, Any

import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum
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

@Model.register("decomposable_attentionModified")
class DecomposableAttentionModified(Model):
    """
    This ``Model`` implements the Decomposable Attention model described in `"A Decomposable
    Attention Model for Natural Language Inference"
    <https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Languag-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27>`_
    by Parikh et al., 2016, with some optional enhancements before the decomposable attention
    actually happens.  Parikh's original model allowed for computing an "intra-sentence" attention
    before doing the decomposable entailment step.  We generalize this to any
    :class:`Seq2SeqEncoder` that can be applied to the premise and/or the hypothesis before
    computing entailment.

    The basic outline of this model is to get an embedded representation of each word in the
    premise and hypothesis, align words between the two, compare the aligned phrases, and make a
    final entailment decision based on this aggregated comparison.  Each step in this process uses
    a feedforward network to modify the representation.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    attend_feedforward : ``FeedForward``
        This feedforward network is applied to the encoded sentence representations before the
        similarity matrix is computed between words in the premise and words in the hypothesis.
    similarity_function : ``SimilarityFunction``
        This is the similarity function used when computing the similarity matrix between words in
        the premise and words in the hypothesis.
    compare_feedforward : ``FeedForward``
        This feedforward network is applied to the aligned premise and hypothesis representations,
        individually.
    aggregate_feedforward : ``FeedForward``
        This final feedforward network is applied to the concatenated, summed result of the
        ``compare_feedforward`` network, and its output is used as the entailment class logits.
    premise_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the premise, we can optionally apply an encoder.  If this is ``None``, we
        will do nothing.
    hypothesis_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the hypothesis, we can optionally apply an encoder.  If this is ``None``,
        we will use the ``premise_encoder`` for the encoding (doing nothing if ``premise_encoder``
        is also ``None``).
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 attend_feedforward: FeedForward,
                 similarity_function: SimilarityFunction,
                 compare_feedforward: FeedForward,
                 aggregate_feedforward: FeedForward,
                 premise_encoder: Optional[Seq2SeqEncoder] = None,
                 hypothesis_encoder: Optional[Seq2SeqEncoder] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 similarity_weight: int = 30) -> None:
        super(DecomposableAttentionModified, self).__init__(vocab, regularizer)
        
        self.label_map = vocab.get_token_to_index_vocabulary('labels')

        label_map = [None]*len(self.label_map)
        for lb,lb_idx in self.label_map.items():
            label_map[lb_idx] = lb
        self.label_map = label_map

        self._text_field_embedder = text_field_embedder
        self._attend_feedforward = TimeDistributed(attend_feedforward)
        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        self._compare_feedforward = TimeDistributed(compare_feedforward)
        self._aggregate_feedforward = aggregate_feedforward
        self._premise_encoder = premise_encoder
        self._hypothesis_encoder = hypothesis_encoder or premise_encoder

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        check_dimensions_match(text_field_embedder.get_output_dim(), attend_feedforward.get_input_dim(),
                               "text field embedding dim", "attend feedforward input dim")
        check_dimensions_match(aggregate_feedforward.get_output_dim(), self._num_labels,
                               "final output dimension", "number of labels")

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

        self.lambda_layer = nn.Sequential(nn.Linear(16, 1,bias=False), MyActivationFunction())

        self.lambda_layer[0].weight.data = torch.tensor([[0.1,0.5,0.5,0.5, 0.5,0.1,0.5,0.5, 0.5,0.5,0.1,0.5, 0.5,0.5,0.5,0.9]])
        self.similarity_weight = similarity_weight        

    def createNERFeatVecAndSimMat(self, p, h, metadata, simWeight):
        numOfSamples, maxLenP = p['tokens'].shape
        numOfSamples, maxLenH = h['tokens'].shape
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
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            From a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional, (default = None)
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
        
        NER_Features,symbolic_similarity_mat = self.createNERFeatVecAndSimMat(premise,hypothesis,metadata,self.similarity_weight)

        #print(NER_Features.shape)
        #print(symbolic_similarity_mat.shape)

        float_NER = NER_Features.float()

        lambda_probabilities = self.lambda_layer(float_NER)
        lambda_probabilities = (torch.transpose(lambda_probabilities,1,2)).reshape((symbolic_similarity_mat.shape[0],
                                                                                    symbolic_similarity_mat.shape[1],
                                                                                    symbolic_similarity_mat.shape[2]))


        if self._premise_encoder:
            embedded_premise = self._premise_encoder(embedded_premise, premise_mask)
        if self._hypothesis_encoder:
            embedded_hypothesis = self._hypothesis_encoder(embedded_hypothesis, hypothesis_mask)

        projected_premise = self._attend_feedforward(embedded_premise)
        projected_hypothesis = self._attend_feedforward(embedded_hypothesis)
        # Shape: (batch_size, premise_length, hypothesis_length)
        similarity_matrix = self._matrix_attention(projected_premise, projected_hypothesis)
        similarity_matrix1 = similarity_matrix 
        float_symbolic = symbolic_similarity_mat.float()

        similarity_matrix = torch.add(torch.mul(lambda_probabilities, similarity_matrix), torch.mul(-1 * torch.sub(lambda_probabilities, 1),float_symbolic))

    
        # Shape: (batch_size, premise_length, hypothesis_length)
        p2h_attention = masked_softmax(similarity_matrix, hypothesis_mask)
        # Shape: (batch_size, premise_length, embedding_dim)
        attended_hypothesis = weighted_sum(embedded_hypothesis, p2h_attention)

        # Shape: (batch_size, hypothesis_length, premise_length)
        h2p_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
        # Shape: (batch_size, hypothesis_length, embedding_dim)
        attended_premise = weighted_sum(embedded_premise, h2p_attention)

        premise_compare_input = torch.cat([embedded_premise, attended_hypothesis], dim=-1)
        hypothesis_compare_input = torch.cat([embedded_hypothesis, attended_premise], dim=-1)

        compared_premise = self._compare_feedforward(premise_compare_input)
        compared_premise = compared_premise * premise_mask.unsqueeze(-1)
        # Shape: (batch_size, compare_dim)
        compared_premise = compared_premise.sum(dim=1)

        compared_hypothesis = self._compare_feedforward(hypothesis_compare_input)
        compared_hypothesis = compared_hypothesis * hypothesis_mask.unsqueeze(-1)
        # Shape: (batch_size, compare_dim)
        compared_hypothesis = compared_hypothesis.sum(dim=1)

        aggregate_input = torch.cat([compared_premise, compared_hypothesis], dim=-1)
        label_logits = self._aggregate_feedforward(aggregate_input)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)
        
        label_map = [self.label_map] * len(label_probs)
        lambda_layer_learnt_weights = [self.lambda_layer[0].weight.data]*len(label_probs)
        output_dict = {"label_logits": label_logits,
                       "label_probs": label_probs,
                       "label_names":label_map,
                       "lambda_layer_learnt_weights":lambda_layer_learnt_weights,
                       "h2p_attention": h2p_attention,
                       "p2h_attention": p2h_attention,
                       "eij":similarity_matrix1,
                       "finaleij":similarity_matrix,
                       "lambdaProb":lambda_probabilities}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["premise_tokens"] = [x["premise_tokens"] for x in metadata]
            output_dict["hypothesis_tokens"] = [x["hypothesis_tokens"] for x in metadata]

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'accuracy': self._accuracy.get_metric(reset),
                }
