from datasets import load_dataset

ds = load_dataset("csv", data_files={
    "train":"dreaddit-train.csv",
    "test":"dreaddit-test.csv"
})

ds.push_to_hub("hutchii/Dreaddit", private=True)