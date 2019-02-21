# Drifter

``` bash
# Run bash in docker container
docker run -it -u root -v $(pwd):/app/codalab codalab/codalab-legacy:py3 bash
cd app/codalab

docker run -d -it -u root -v $(pwd):/app/codalab codalab/codalab-legacy:py3 bash
docker exec -it <container_name> bin/bash

docker exec -it laughing_haibt bin/bash

docker exec -it fervent_villani bin/bash

# For ingestion (Sample Data) 
python3 AutoML3_ingestion_program/ingestion.py AutoML3_sample_data AutoML3_sample_predictions AutoML3_sample_data AutoML3_ingestion_program rafiki_lifelong

python3 AutoML3_ingestion_program/ingestion.py AutoML3_sample_data AutoML3_sample_predictions AutoML3_sample_data AutoML3_ingestion_program Drifter

nohup python3 AutoML3_ingestion_program/ingestion.py AutoML3_sample_data AutoML3_sample_predictions AutoML3_sample_data AutoML3_ingestion_program rafiki_lifelong &

# For scoring (Sample Data)
python3 AutoML3_scoring_program/score.py 'AutoML3_sample_data/*/' AutoML3_sample_predictions AutoML3_scoring_output

# For ingestion (Actual Data)
python3 AutoML3_ingestion_program/ingestion.py AutoML3_input_data AutoML3_input_predictions AutoML3_input_data AutoML3_ingestion_program rafiki_lifelong

python3 AutoML3_ingestion_program/ingestion.py AutoML3_input_data AutoML3_input_predictions AutoML3_input_data AutoML3_ingestion_program Drifter

nohup python3 AutoML3_ingestion_program/ingestion.py AutoML3_input_data AutoML3_input_predictions AutoML3_input_data AutoML3_ingestion_program Drifter &

nohup python3 AutoML3_ingestion_program/ingestion.py AutoML3_input_data AutoML3_input_predictions AutoML3_input_data AutoML3_ingestion_program rafiki_lifelong &

# For scoring (Actual Data)
python3 AutoML3_scoring_program/score.py 'AutoML3_input_data/*/' AutoML3_input_predictions AutoML3_scoring_output

ps ax | grep AutoML3
```