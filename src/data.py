import argparse
import kagglehub
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("dataset_input", type=str, help='path to whole datset') # don't think it's needed with kaggle api
    parser.add_argument("all_train_output", type=str, help='path to cleaned whole dataset')
    parser.add_argument("train_output_1", type=str, help='path to first training input .tsv file')
    parser.add_argument("train_output_2", type=str, help='path to second training input .tsv file')
    parser.add_argument("train_output_3", type=str, help='path to third training input .tsv file')
    parser.add_argument("train_output_4", type=str, help='path to fourth training input .tsv file')
    parser.add_argument("test_output", type=str, help='path to test input .tsv file')
    args = parser.parse_args()

    # TODO: Remove unwanted columns, reserve ~10% for testing, randomly divide rest into four subgroups

    # kaggle download to df
    path = kagglehub.dataset_download("sacramentotechnology/sleep-deprivation-and-cognitive-performance")
    file_path = os.path.join(path, "sleep_deprivation_dataset_detailed.csv")
    df = pd.read_csv(file_path)
    #print(df.head)

    # remove unwanted columns
    unwanted_columns = ["Daytime_Sleepiness", "Stroop Task reaction time", "N-back accuracy", "PVT reaction time"]
    df_clean = df.drop(columns=[col for col in unwanted_columns if col in df.columns])
    
    df_clean.to_csv(args.all_train_output, sep='\t', index=False)
    
    # reserve 10% of the data for testing and use the remaining for training
    train_df, test_df = train_test_split(df_clean, test_size=0.1, random_state=42)
    
    # shuffle the training set and split it into four roughly equal subgroups
    train_df_shuffled = train_df.sample(frac=1, random_state=42)
    train_groups = np.array_split(train_df_shuffled, 4)
    
    train_groups[0].to_csv(args.train_output_1, sep='\t', index=False)
    train_groups[1].to_csv(args.train_output_2, sep='\t', index=False)
    train_groups[2].to_csv(args.train_output_3, sep='\t', index=False)
    train_groups[3].to_csv(args.train_output_4, sep='\t', index=False)
    
    # Save the test set to file (TSV format)
    test_df.to_csv(args.test_output, sep='\t', index=False)
    
    print("Data processing complete. Cleaned dataset, training subsets, and test set have been saved.")
