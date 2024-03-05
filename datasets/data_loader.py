import pickle
from catenets.datasets import load as catenets_load
from news.process_news import process_news
from tcga.process_tcga import process_tcga

<<<<<<< HEAD

=======
>>>>>>> 72bf8dc (First commit)
def load(dataset_name: str, train_ratio: float = 1.0):
    if "tcga" in dataset_name:
        try:
            tcga_dataset = pickle.load(
                open(
<<<<<<< HEAD
                    "tcga/" + str(dataset_name) + ".p",
=======
                    "datasets/tcga/" + str(dataset_name) + ".p",
>>>>>>> 72bf8dc (First commit)
                    "rb",
                )
            )
        except:
            process_tcga(
<<<<<<< HEAD
                max_num_genes=100, file_location="tcga/"
            )
            tcga_dataset = pickle.load(
                open(
                    "tcga/" + str(dataset_name) + ".p",
=======
                max_num_genes=100, file_location="datasets/tcga/"
            )
            tcga_dataset = pickle.load(
                open(
                    "datasets/tcga/" + str(dataset_name) + ".p",
>>>>>>> 72bf8dc (First commit)
                    "rb",
                )
            )
        X_raw = tcga_dataset["rnaseq"]
    elif "news" in dataset_name:
        try:
            news_dataset = pickle.load(
                open(
<<<<<<< HEAD
                    "news/" + str(dataset_name) + ".p",
=======
                    "datasets/news/" + str(dataset_name) + ".p",
>>>>>>> 72bf8dc (First commit)
                    "rb",
                )
            )
        except:
            process_news(
<<<<<<< HEAD
                max_num_features=100, file_location="news/"
            )
            news_dataset = pickle.load(
                open(
                    "news/" + str(dataset_name) + ".p",
=======
                max_num_features=100, file_location="datasets/news/"
            )
            news_dataset = pickle.load(
                open(
                    "datasets/news/" + str(dataset_name) + ".p",
>>>>>>> 72bf8dc (First commit)
                    "rb",
                )
            )
        X_raw = news_dataset
    elif "twins" in dataset_name:
        # Total features  = 39
        X_raw, _, _, _, _, _ = catenets_load(dataset_name, train_ratio=1.0)
    elif "acic" in dataset_name:
        # Total features  = 55
        X_raw, _, _, _, _, _, _, _ = catenets_load("acic2016")
    else:
        print("Unknown dataset " + str(dataset_name))

    if train_ratio == 1.0:
        return X_raw
    else:
        X_raw_train = X_raw[: int(train_ratio * X_raw.shape[0])]
        X_raw_test = X_raw[int(train_ratio * X_raw.shape[0]) :]
        return X_raw_train, X_raw_test
