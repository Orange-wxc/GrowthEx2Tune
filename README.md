# **GrowthEx2Tune: An Intelligent Buffer Tuning Method for Decoupled Storage-Compute Database Clusters**

## **Code Structure**
- **test_model:** The results of the tuning process will be placed here after tuning.
- **tune:** Includes the training code necessary for the tuning process.
- **my_algorithm:** Houses the agent, deep reinforcement learning (DRL) algorithms, and model code.
- **maEnv:** Provides ancillary code supportive of the decoupled storage-compute database architecture.
- **main.py:** The primary script used to initiate the tuning process.
- **globalValue.py:** Configuration file for setting up the GrowthEx2Tune.
- **data:** Data for sysbench and tpcc-mysql.

## **Environment Setup**
- **Operating System:** Ubuntu 18.04.6
- **Python Version:** 3.7.9
- **Additional Libraries:** Parl 1.3.1
- **Dependencies:** A detailed list of required Python packages is available in `requirements.txt`.

## **Running the Tool**
To initiate the tuning process, ensure that the status monitor, action receiver, and executor are correctly implemented and operational within your database. Start the process using the command below:

```bash
python main.py
```

## **Other Notes**
- **EKC Rules:** You can customise your own labels and rules in 
`maEnv/datautils.py/status_to_labels` and `maEnv/datautils.py/node_labels_to_action_trend`.
- **Hyperparameters:** In order to get a good effect of tuning, you should you should adjust the hyperparameters in `tune/train_wxc.py` appropriately.
