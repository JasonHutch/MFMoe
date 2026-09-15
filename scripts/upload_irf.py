from datasets import load_dataset

ds = load_dataset("csv", data_files={
    "train":"/Users/hutchii/PycharmProjects/MFMoE/data/irf/train_data.csv",
    "test":"/Users/hutchii/PycharmProjects/MFMoE/data/irf/test_data.csv"
})

ds.push_to_hub("hutchii/InterpersonalRiskFactors", private=True)