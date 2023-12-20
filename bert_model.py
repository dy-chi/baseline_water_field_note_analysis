import sqlite3
import pandas as pd
# Connect to the SQLite database
db_path = r'C:\Users\dychi\Alberta Water Wells\bbwt.db'

# Replace with your SQLite database path
conn = sqlite3.connect(db_path)

# SQL query to select all columns from the 'GAS_ANALYSIS_INFO' table
query = "SELECT * FROM FIELD_NOTES;"

# Create a DataFrame from the query result
df_field_notes = pd.read_sql_query(query, conn)

# Close the connection
conn.close()

# Display the first few rows of the DataFrame
print(df_field_notes.head())

# create new df with manually labelled data in the odour category column. 
# 1 means odour not mentioned
# 2 means odour is good
# 3 means there is "fair" odour or otherwise noticible odour to water
# 0 means there was an ü in the field notes comments, meaning the comment is in a "form" format and therefore not appicable to EDA 
df_labelled = df_field_notes[pd.notna(df_field_notes['ODOUR_CAT'])]

df_labelled['ODOUR_CAT'].hist()

# Drop rows with 'ODOUR_CAT' values 0 and 1 (values where odour is not mentioned (as asssessed manually): 1, or there is a checklist: 0)

# 3 means there was an odour, and 2 means there was no odour, or the odour was decribed as good

df_labelled = df_labelled[~df_labelled['ODOUR_CAT'].isin([0, 1])]

# Replace values in 'ODOUR_CAT' column
df_labelled['ODOUR_CAT'] = df_labelled['ODOUR_CAT'].replace({3: 1, 2: 0})

# replace null values with 0, 0 meaning H2S was not tested for 
df_field_notes['H2S_TESTED'] = df_field_notes['H2S_TESTED'].fillna(0)

import pandas as pd
from sklearn.utils import resample

# function to resample labelled data

def random_undersample(df, target_column, minority_class_value, majority_class_value, undersample_ratio=1.0, random_seed=None):
    """
    Randomly undersample a binary categorical feature in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target_column (str): The column to undersample.
    - minority_class_value: The value representing the minority class.
    - majority_class_value: The value representing the majority class.
    - undersample_ratio (float): The ratio of the number of minority class samples to keep.
    - random_seed (int or None): Seed for reproducibility.

    Returns:
    - pd.DataFrame: The undersampled DataFrame.
    """

    # Separate majority and minority classes
    majority_class = df[df[target_column] == majority_class_value]
    minority_class = df[df[target_column] == minority_class_value]

    # Undersample majority class
    undersampled_majority = resample(
        majority_class,
        replace=False,
        n_samples=int(len(minority_class) * undersample_ratio),
        random_state=random_seed
    )

    # Concatenate minority class and undersampled majority class
    undersampled_df = pd.concat([minority_class, undersampled_majority])

    return undersampled_df

undersampled_labelled=random_undersample(df_labelled, 'ODOUR_CAT', 1, 0, undersample_ratio=1.0, random_seed=9898)


#initiate arrays for model
text = undersampled_labelled.FIELD_NOTE_COMMENTS.values
labels = undersampled_labelled['ODOUR_CAT'].values

from transformers import BertTokenizer

# Download the tokenizer for 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Save the tokenizer to a local directory (optional)
tokenizer.save_pretrained('bert-base-uncased-tokenizer')

# You can now use the 'tokenizer' object to tokenize and encode text

#preprocess data for model

token_id = []
attention_masks = []

def preprocessing(input_text, tokenizer):
  '''
  Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
  '''
  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = 64,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt'
                   )


for sample in text:
  encoding_dict = preprocessing(sample, tokenizer)
  token_id.append(encoding_dict['input_ids']) 
  attention_masks.append(encoding_dict['attention_mask'])


token_id = torch.cat(token_id, dim = 0)
attention_masks = torch.cat(attention_masks, dim = 0)

labels = torch.tensor(labels)
labels = labels.long()


val_ratio = 0.2
# Recommended batch size: 16, 32. See: https://arxiv.org/pdf/1810.04805.pdf
batch_size = 16

# Indices of the train and validation splits stratified by labels
train_idx, val_idx = train_test_split(
    np.arange(len(labels)),
    test_size = val_ratio,
    shuffle = True,
    stratify = labels)

# Train and validation sets
train_set = TensorDataset(token_id[train_idx], 
                          attention_masks[train_idx], 
                          labels[train_idx])

val_set = TensorDataset(token_id[val_idx], 
                        attention_masks[val_idx], 
                        labels[val_idx])

# Prepare DataLoader
train_dataloader = DataLoader(
            train_set,
            sampler = RandomSampler(train_set),
            batch_size = batch_size
        )

validation_dataloader = DataLoader(
            val_set,
            sampler = SequentialSampler(val_set),
            batch_size = batch_size
        )


def b_tp(preds, labels):
    '''Returns True Positives (TP): count of correct predictions of actual class 1'''
    return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_fp(preds, labels):
    '''Returns False Positives (FP): count of wrong predictions of actual class 1'''
    return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_tn(preds, labels):
    '''Returns True Negatives (TN): count of correct predictions of actual class 0'''
    return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_fn(preds, labels):
    '''Returns False Negatives (FN): count of wrong predictions of actual class 0'''
    return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])


def b_metrics(preds, labels):
    '''Returns the following metrics:
    - accuracy    = (TP + TN) / N
    - precision   = TP / (TP + FP)
    - recall      = TP / (TP + FN)
    - specificity = TN / (TN + FP)'''
    preds = np.argmax(preds, axis = 1).flatten()
    labels = labels.flatten()
    tp = b_tp(preds, labels)
    tn = b_tn(preds, labels)
    fp = b_fp(preds, labels)
    fn = b_fn(preds, labels)
    b_accuracy = (tp + tn) / len(labels)
    b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
    b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
    b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
    return b_accuracy, b_precision, b_recall, b_specificity


# Load the BertForSequenceClassification model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
)

# Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr = 5e-5,
                              eps = 1e-08
                              )

# Run on GPU
#model.cuda()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Recommended number of epochs: 2, 3, 4. See: https://arxiv.org/pdf/1810.04805.pdf
epochs = 2

for _ in trange(epochs, desc = 'Epoch'):
    
    # ========== Training ==========
    
    # Set model to training mode
    model.train()
    
    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        # Forward pass
        train_output = model(b_input_ids, 
                             token_type_ids = None, 
                             attention_mask = b_input_mask, 
                             labels = b_labels)
        # Backward pass
        train_output.loss.backward()
        optimizer.step()
        # Update tracking variables
        tr_loss += train_output.loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    # ========== Validation ==========

    # Set model to evaluation mode
    model.eval()

    # Tracking variables 
    val_accuracy = []
    val_precision = []
    val_recall = []
    val_specificity = []

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
          # Forward pass
          eval_output = model(b_input_ids, 
                              token_type_ids = None, 
                              attention_mask = b_input_mask)
        logits = eval_output.logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Calculate validation metrics
        b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
        val_accuracy.append(b_accuracy)
        # Update precision only when (tp + fp) !=0; ignore nan
        if b_precision != 'nan': val_precision.append(b_precision)
        # Update recall only when (tp + fn) !=0; ignore nan
        if b_recall != 'nan': val_recall.append(b_recall)
        # Update specificity only when (tn + fp) !=0; ignore nan
        if b_specificity != 'nan': val_specificity.append(b_specificity)

    print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
    print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
    print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
    print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
    print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity)/len(val_specificity)) if len(val_specificity)>0 else '\t - Validation Specificity: NaN')

