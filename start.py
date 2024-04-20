X_numpy = X_tensor.numpy()
y_numpy = y_tensor.squeeze(1).numpy()  # Squeeze the extra dimension

# Define the number of splits for Stratified K-Fold cross-validation
num_splits = 5

# Initialize Stratified K-Fold cross-validator
skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

# Create dictionaries to store metrics for each fold
fold_metrics = {
    'train_loss': [],
    'val_loss': [],
    'train_accuracy': [],
    'val_accuracy': []
}

t0 = time()  # Record the starting time for K-Fold cross-validation
# Iterate over folds
for fold, (train_indices, val_indices) in enumerate(skf.split(X_numpy, y_numpy)):

    print(f'\nFold {fold + 1}/{num_splits}')

    # Split data into training and validation sets
    X_train_fold, X_val_fold = torch.FloatTensor(X_numpy[train_indices]), torch.FloatTensor(X_numpy[val_indices])
    y_train_fold, y_val_fold = torch.FloatTensor(y_numpy[train_indices]).unsqueeze(1), torch.FloatTensor(
        y_numpy[val_indices]).unsqueeze(1)

    # Create datasets and data loaders for training and validation sets
    train_dataset_fold = TensorDataset(X_train_fold, y_train_fold)
    val_dataset_fold = TensorDataset(X_val_fold, y_val_fold)

    train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)
    val_loader_fold = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = models.CNN_BinaryClassifier(input_channels=in_channels, kernels=kernels, kernel_size=kernel_size,
                                        input_features=in_features, classes=classes).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize lists to store metrics for this fold
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Iterate over batches in the training set
        for inputs, labels in train_loader_fold:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader_fold)
        accuracy = correct_predictions / total_samples

        # if (epoch + 1) % 5 == 0:
        print(f'Training - Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')

        # Append metrics to lists
        train_losses.append(average_loss)
        train_accuracies.append(accuracy)

        # Validation loop
        model.eval()

        val_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0

        # Disable gradient calculation during validation
        with torch.no_grad():
            # Iterate over batches in the validation set
            for val_inputs, val_labels in val_loader_fold:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()

                # Calculate accuracy
                val_predicted = (val_outputs > 0.5).float()
                val_correct_predictions += (val_predicted == val_labels).sum().item()
                val_total_samples += val_labels.size(0)

            val_average_loss = val_loss / len(val_loader_fold)
            val_accuracy = val_correct_predictions / val_total_samples

            print(
                f'Validation - Epoch [{epoch + 1}/{num_epochs}], Loss: {val_average_loss:.4f}, Accuracy: {val_accuracy * 100:.2f}%')

            # Append metrics to lists
            val_losses.append(val_average_loss)
            val_accuracies.append(val_accuracy)

    # Store metrics for this fold in the dictionary
    fold_metrics['train_loss'].append(train_losses)
    fold_metrics['val_loss'].append(val_losses)
    fold_metrics['train_accuracy'].append(train_accuracies)
    fold_metrics['val_accuracy'].append(val_accuracies)

t1 = time()  # Record the ending time for K-Fold cross-validation

# Calculate and print the time taken for K-Fold cross-validation of the model
preprocessing_d["Time <K-Fold cross validation>"] = t1 - t0
print(f"Time <K-Fold cross validation>: {preprocessing_d['Time <K-Fold cross validation>']:.3f} seconds")