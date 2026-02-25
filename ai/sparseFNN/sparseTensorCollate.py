import torch

def sparse_collate_fn(batch):
    # 'batch' is a list of samples from the Dataset, where each sample 
    # might be a tuple of (sparse_features, label)
    #sample(features,label)
    # Separate features and labels
    features_list, labels_list = zip(*batch) 
    
    # Process features: Combine individual sparse tensors into one batched sparse tensor
    # The key is adjusting indices to account for the batch dimension
    
    batched_indices = []
    batched_values = []
    
    # Track the cumulative size of the batch dimension
    batch_size = len(features_list)
    max_dim_0 = 0 # Assuming features are 2D (or higher, but we focus on batching dim 0)
    
    for i, features in enumerate(features_list):
        # features is a sparse tensor
        indices = features.indices().clone()
        values = features.values()
        
        # Shift indices by the current batch index
        # This assumes the features have a shape like (N_i, Feature_dim)
        # We add 'i' to the first dimension of all indices
        indices[0] = indices[0] + i
        
        batched_indices.append(indices)
        batched_values.append(values)
        if features.shape[0] > max_dim_0:
            max_dim_0 = features.shape[0]

    # Concatenate indices and values
    # If the original features are sparse tensors, convert them to dense tensors
    # and stack into a batched dense feature tensor (float).
    # Also convert labels to dense and keep them as Long for loss functions that expect integer targets.
    all_indices = torch.cat(batched_indices, dim=1) if len(batched_indices) > 0 else torch.tensor([])
    all_values = torch.cat(batched_values, dim=0) if len(batched_values) > 0 else torch.tensor([])

    # Attempt to reconstruct dense feature and label tensors from the original batch
    try:
        dense_features = torch.stack([f.to_dense().view(-1) for f in features_list]).float()
        dense_labels = torch.stack([l.to_dense().view(-1) for l in labels_list]).float()
        return dense_features, dense_labels
    except Exception:
        # Fallback: return the raw concatenated indices/values if conversion fails
        return all_indices, all_values

    