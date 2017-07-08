
export BUCKET_NAME=sahib-tensorflow
export JOB_NAME="example_5_train_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://$BUCKET_NAME/$JOB_NAME --runtime-version 1.0 --module-name trainer.example5 --package-path ./trainer --config=trainer/cloudml-gpu.yaml -- --train-file gs://sahib-tensorflow/train.csv
