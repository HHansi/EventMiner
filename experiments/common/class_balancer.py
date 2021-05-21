# Created by Hansi at 5/6/2021


def binary_class_balance(df, label_column, label, proportion=0.75):
    """
    Reduce the proportion of given label instances to given proportion
    :param df: dataframe
    :param label_column: str
        name of the label column
    :param label: e.g. 1 or 0
    :param proportion: float
        proportion need to be assigned to the given label in whole data set
    :return:
    """
    if proportion >= 1:
        raise ValueError('Proportion need to be < 1!')
    unique_labels = list(df[label_column].unique())
    unique_labels.remove(label)
    other_label = unique_labels[0]

    label_df = df[df[label_column] == label]
    other_df = df[df[label_column] == other_label]

    other_count = len(other_df)
    required_label_count = int((other_count / (1 - proportion)) * proportion)

    filtered = label_df.head(required_label_count)  # filter data
    new_df = other_df.append(filtered)  # combine data
    new_df = new_df.sample(frac=1).reset_index(drop=True)  # shuffle data

    # print(f'label proportion: {len(new_df[new_df[label_column]==label])/len(new_df)}')
    # print(f'other proportion: {len(new_df[new_df[label_column] == other_label]) / len(new_df)}')
    return new_df


