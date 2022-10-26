import argparse
from attoDNN.attodataset import AttoDataset
from attoDNN.prepare_data import dataset_to_CEP_averaged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Take CEP average of a dataset")
    parser.add_argument("input", help="Input file path.")
    parser.add_argument("output", help="Output file path.")
    args = parser.parse_args()


    dataset = AttoDataset(args.input)

    dataset_to_CEP_averaged(dataset, args.output)
