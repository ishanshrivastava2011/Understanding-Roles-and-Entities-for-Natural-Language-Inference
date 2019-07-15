
from overrides import overrides
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.predictors.predictor import Predictor
import numpy as np
from allennlp.models import Model
import json
@Predictor.register('nli_predictor')
class nliPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.DecomposableAttention` model.
    """

    def __init__(self,model: Model, dataset_reader: DatasetReader):
        self._numOfInstances = 0
        self._numCorrectPredictions = 0
        self._currGoldLabel = None
        self._model = model
        self._dataset_reader = dataset_reader


    def predict(self, premise: str, hypothesis: str) -> JsonDict:
        """
        Predicts whether the hypothesis is entailed by the premise text.

        Parameters
        ----------
        premise : ``str``
            A passage representing what is assumed to be true.

        hypothesis : ``str``
            A sentence that may be entailed by the premise.

        Returns
        -------
        A dictionary where the key "label_probs" determines the probabilities of each of
        [entailment, contradiction, neutral].
        """
        return self.predict_json({"premise" : premise, "hypothesis": hypothesis})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"premise": "...", "hypothesis": "..."}``.
        """
        premise_text = json_dict["premise"]
        hypothesis_text = json_dict["hypothesis"]
        print("My Json To Instance")
        print(json_dict)
        self._currGoldLabel = json_dict["gold_label"]
        return self._dataset_reader.text_to_instance(premise_text, hypothesis_text)

    #@overrides
    #def predict_instance(self, instance: Instance) -> JsonDict:
    #   outputs = self._model.forward_on_instance(instance)
    #    return outputs

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        #ISHAN - Added print statment to debug
        print(outputs["label_probs"])
        self._numOfInstances+=1
        print(self._numOfInstances)
        outputs['AccuracySoFar'] = self.calAccEVsNotE(outputs)
        return json.dumps(outputs) + "\n"
    #    #return json.dumps(outputs) + "\n"
    
    def calAccEVsNotE(self, outputs):
        labels = ['entailment', 'contradiction', 'neutral']
        a = np.argmax(outputs["label_probs"]) 
        predictedLabel = labels[a]
        print(predictedLabel)
        print(self._currGoldLabel)
        if predictedLabel == 'entailment' and self._currGoldLabel  == 'entailment':
            self._numCorrectPredictions+=1
        elif predictedLabel != 'entailment' and self._currGoldLabel  != 'entailment':
            self._numCorrectPredictions+=1
        return self._numCorrectPredictions/self._numOfInstances
           
