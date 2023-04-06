import argparse
import pickle
import collections.abc


def deep_update_dict(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    for key, value in overrides.items():
        if isinstance(value, collections.abc.Mapping) and value:
            returned = deep_update_dict(source.get(key, {}), value)
            source[key] = returned
        else:
            if key in source.keys():
                print('Overriding ')
                print(key)
            source[key] = overrides[key]
    return source


def append_unique_list(source, overrides):
    for x in overrides:
        if x not in source:
            source.append(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Join multiple analysis.pkl produced by evaluate.py when they have different training/testing datasets and/or different models.')
    parser.add_argument('output_pickle', type=str, help='File to save pickle results of joining the analyses.')
    parser.add_argument('analyses', type=str, nargs='+', help='List of analysis.pkl to join.')

    args = parser.parse_args()
    
    grid = {}
    unique_model_names = []
    unique_train_dataset_names = []
    unique_eval_dataset_names = []
    
    
    for analysis in args.analyses:
        with open(analysis, 'rb') as f:
            grid1, unique_model_names1, unique_train_dataset_names1, unique_eval_dataset_names1 = pickle.load(f)
            
            deep_update_dict(grid, grid1)
            append_unique_list(unique_model_names, unique_model_names1)
            append_unique_list(unique_train_dataset_names, unique_train_dataset_names1)
            append_unique_list(unique_eval_dataset_names, unique_eval_dataset_names1)
    
    print('Models')
    for model in unique_model_names:
        print(model)
    
    print('Train datasets')
    for ds in unique_train_dataset_names:
        print(ds)
        
    print('Evaluation datasets')
    for ds in unique_eval_dataset_names:
        print(ds)
    
    with open(args.output_pickle, 'wb') as f:
        pickle.dump((grid, unique_model_names,
                     unique_train_dataset_names,
                     unique_eval_dataset_names), f)
        print(f'Analysis saved to {args.output_pickle}.')
