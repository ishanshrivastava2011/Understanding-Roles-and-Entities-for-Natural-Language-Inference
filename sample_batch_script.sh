source activate allennlp_editable

config_path=./new_experiments_iter2/training_config/esim_lambda_snli_nc_rs.jsonnet
serialization_dir=./new_experiments_iter2/ModifiedESIM/TrainOnSNLI_NC_RS/Glove/
test_data_path=./test_data/

model=model.tar.gz
evaluation_list=($test_data_path"snliTest.txt" $serialization_dir"testEvaluate_snli.txt" $test_data_path"NC_test.jsonl" $serialization_dir"testEvaluate_NC.txt" $test_data_path"RS_test.jsonl" $serialization_dir"testEvaluate_RS.txt")
make_predictions_list=($test_data_path"NC_test.jsonl" $serialization_dir"testPredict_NC.txt" $test_data_path"RS_test.jsonl" $serialization_dir"testPredict_RS.txt")

mkdir -p $serialization_dir

model_path=$serialization_dir$model


echo "Config File :: $config_path"
echo "Serialization Directory :: $serialization_dir"
echo "Test Data Path :: $test_data_path"
echo "Evaluation List :: $evaluation_list"
echo "Predictios List :: $make_predictions_list"

echo "Training..."
allennlp train $config_path --serialization-dir $serialization_dir --include-package allennlp.my_library

echo "Evaluating.."
for ((i=0; i<${#evaluation_list[@]}; i+=2)); do
    echo "Evaluating for :: ${evaluation_list[i]} and saving the evaluations in :: ${evaluation_list[i+1]}"
    allennlp evaluate $model_path ${evaluation_list[i]} --include-package allennlp.my_library --output-file ${evaluation_list[i+1]} --cuda-device 0
done

echo "Predicting.."
for ((i=0; i<${#make_predictions_list[@]}; i+=2)); do
    echo "Predicting for :: ${make_predictions_list[i]} and saving the precictions in :: ${make_predictions_list[i+1]}"
    allennlp predict $model_path ${make_predictions_list[i]} --predictor textual-entailment --output-file ${make_predictions_list[i+1]}  --use-dataset-reader --cuda-device 0 --include-package allennlp.my_library
done

