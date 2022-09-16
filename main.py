"""
Author: Elias Mindlberger
Matr.Nr.: 12043382
Exercise 5

    The functions below are designed to train a Neural Network to in-paint images.

"""
import numpy as np
import torch
from datasets import Images, TransformedImages
import torchvision.transforms as transforms
import torch.utils.data
from architectures import CNN
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from tqdm import tqdm
import os


def main(data_path: str = os.path.join("..", "training"), epochs: int = 30_000, learning_rate: float = 5e-4,
         optimizer: str = "Adam", weight_decay: float = 5e-5, normalize: bool = True, crop: bool = True,
         im_shape: int = 100, batch_size: int = 32, results_path: str = "results", plot_path: str = "results/plots"):
    """
    Main function for training the model
    """
    # Set a known random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Prepare a path to plot to
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

    # get data
    dataset = Images(image_directory=data_path, normalize=normalize, crop=crop, im_shape=im_shape)
    # get device
    device = _get_device()

    # Split the dataset
    # allocate 70 % to training, 15 % to validation, 15 % to testing
    trainingset = torch.utils.data.Subset(
        dataset,
        indices=np.arange(int(len(dataset) * 0.8))
    )
    validationset = torch.utils.data.Subset(
        dataset,
        indices=np.arange(int(len(dataset) * 0.1))
    )
    testset = torch.utils.data.Subset(
        dataset,
        indices=np.arange(int(len(dataset) * 0.1))
    )

    # Create Test and Validation Set loaders
    val_set = TransformedImages(images=validationset)
    test_set = TransformedImages(images=testset)
    val_set_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=1, shuffle=False, num_workers=6, collate_fn=_collate
    )
    test_set_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=1, shuffle=False, num_workers=6, collate_fn=_collate
    )

    # Create data loader for the training process
    # Add augmentation to images
    trainingset_augmented = TransformedImages(images=trainingset, transforms_chain=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ]))
    training_loader = torch.utils.data.DataLoader(
        dataset=trainingset_augmented, batch_size=batch_size, shuffle=True, num_workers=6, collate_fn=_collate
    )

    # Initialize CNN
    model = CNN()
    model.to(device)

    # MSE Loss
    mse = torch.nn.MSELoss()

    # Define the optimizer
    optim = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay) if optimizer == "SGD" \
        else torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    plot_interval = 1500
    stats_interval = 100
    validation_interval = 500

    # Save initial model
    saved_model = os.path.join(results_path, "best_model.pt")
    torch.save(model, saved_model)

    # Define Tensorboard Writer
    writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))
    progress_bar = tqdm(total=epochs, desc=f"loss: {np.nan:7.5f}", position=0)

    best_validation_loss = np.inf
    # Specify the training process
    epoch = 1
    while epoch < epochs:
        for data in training_loader:
            inputs_stacked, known_arrays_stacked, targets_tensor, images_stacked = data
            inputs_stacked = inputs_stacked.to(device)
            images_stacked = images_stacked.to(device)

            # Reset grad
            optim.zero_grad()

            # Get output
            output = model(inputs_stacked)

            # Calculate loss - back-prop - step
            loss = mse(output, images_stacked)
            loss.backward()
            optim.step()

            # Print current status and score
            if (epoch + 1) % stats_interval == 0:
                writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(), global_step=epoch)

            # Plot output
            if (epoch + 1) % plot_interval == 0:
                _plot(inputs_stacked.detach().cpu().numpy(),
                      images_stacked.detach().cpu().numpy(),
                      output.detach().cpu().numpy(),
                      plot_path, epoch)

            # Evaluate model on validation set
            if (epoch + 1) % validation_interval == 0:
                val_loss = _evaluate_model(model, data_loader=val_set_loader, loss_fn=mse)
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss, global_step=epoch)
                # Add weights and gradients as arrays to tensorboard
                for i, (name, param) in enumerate(model.named_parameters()):
                    writer.add_histogram(tag=f"validation/param_{i} ({name})",
                                         values=param.cpu(), global_step=epoch)
                    writer.add_histogram(tag=f"validation/gradients_{i} ({name})",
                                         values=param.grad.cpu(), global_step=epoch)
                # Save best model for early stopping
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    torch.save(model, saved_model)

            progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            progress_bar.update()
            epoch += 1
            if epoch >= epochs:
                break

    progress_bar.close()
    writer.close()
    print("Finished Training!")

    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(saved_model)
    train_loss = _evaluate_model(net, data_loader=training_loader, loss_fn=mse)
    val_loss = _evaluate_model(net, data_loader=val_set_loader, loss_fn=mse)
    test_loss = _evaluate_model(net, data_loader=test_set_loader, loss_fn=mse)

    print(f"Scores:")
    print(f"  training loss: {train_loss}")
    print(f"validation loss: {val_loss}")
    print(f"      test loss: {test_loss}")

    # Write result to file
    with open(os.path.join(results_path, "results.txt"), "w") as rf:
        print(f"Scores:", file=rf)
        print(f"  training loss: {train_loss}", file=rf)
        print(f"validation loss: {val_loss}", file=rf)
        print(f"      test loss: {test_loss}", file=rf)


def _get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else AssertionError("Couldn't initialize with CUDA.")


def _evaluate_model(model, data_loader, loss_fn):
    """
        Function for computing the loss over some subset of the data
    """
    device = _get_device()
    # Set the model in evaluation mode
    model.eval()
    loss = 0
    # No gradient-calculations needed
    with torch.no_grad():
        # Loop over the data-loader
        for data in tqdm(data_loader, desc="scoring", position=0):
            # Get a sample
            inputs_stacked, known_arrays_stacked, targets_tensor, images_stacked = data
            inputs_stacked = inputs_stacked.to(device)
            images_stacked = images_stacked.to(device)
            # Get model outputs
            model_outputs = model(inputs_stacked)
            # Add losses up
            loss += loss_fn(model_outputs, images_stacked).item()
    # Divide loss over the number of samples -> average loss over data-subset
    loss /= len(data_loader)
    # Set the model back to training-mode
    model.train()
    return loss


def _plot(inputs, targets, predictions, path, update):
    """Plotting the inputs, targets and predictions to file `path`"""
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

    for i in range(len(inputs)):
        for ax, data, title in zip(axes, [inputs.astype('uint8'), targets.astype('uint8'), predictions.astype('uint8')], ["Input", "Target", "Prediction"]):
            ax.clear()
            ax.set_title(title)
            ax.imshow(np.transpose(data[i], (1, 2, 0))[:, :, 0:3], interpolation="none")
            ax.set_axis_off()
        fig.savefig(os.path.join(path, f"{update:07d}_{i:02d}.png"), dpi=100)

    plt.close(fig)


def _collate(batch_list: list):
    # stack inputs
    inputs_stacked = torch.stack([torch.from_numpy(np.concatenate((data[0], data[1][:1, :, :])))
                                  for data in batch_list])
    # stack knowns
    known_arrays_stacked = torch.stack([torch.from_numpy(data[1]) for data in batch_list])

    # stack target(-images)
    images_stacked = torch.stack([torch.from_numpy(data[3]) for data in batch_list])

    # get the target arrays as a list
    targets = [sample[2] for sample in batch_list]
    # get the maximum length of the target arrays in the current batch
    max_len = np.max([target.shape[0] for target in targets])
    # create a tensor of size (batch_size, max_len) to hold the target arrays
    targets_tensor = torch.zeros(size=(len(targets), max_len, ), dtype=torch.float32)
    # Write the target to the stacked (padded) targets
    for i, target in enumerate(targets):
        targets_tensor[i, :len(target)] = torch.from_numpy(target)
    return inputs_stacked, known_arrays_stacked, targets_tensor, images_stacked
