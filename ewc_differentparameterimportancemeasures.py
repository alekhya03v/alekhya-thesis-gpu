import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset


# ----------------------------
# 1️⃣ Setup device, seeds, hyperparameters
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

n_tasks = 10
hidden_width = 400
learning_rate = 1e-3
momentum = 0.9
epochs = 5
batch_size = 64
ewc_lambda = 0.1


# ----------------------------
# 2️⃣ Data Loading
# ----------------------------
def get_dataloaders(batch_size):
    mnist_dataset = load_dataset("mnist").with_format("torch")

    dataloader_train = torch.utils.data.DataLoader(
        mnist_dataset["train"], batch_size=batch_size, shuffle=True
    )

    dataloader_test = torch.utils.data.DataLoader(
        mnist_dataset["test"], batch_size=batch_size, shuffle=False
    )

    dataloader_fim = torch.utils.data.DataLoader(
        mnist_dataset["train"], batch_size=1, shuffle=True
    )

    return dataloader_train, dataloader_test, dataloader_fim


# ----------------------------
# 3️⃣ Define model
# ----------------------------
def get_model():
    model = nn.Sequential(
        nn.Linear(28 * 28, hidden_width),
        nn.ReLU(),
        nn.Linear(hidden_width, hidden_width),
        nn.ReLU(),
        nn.Linear(hidden_width, 10),
    ).to(device)

    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)

    return model


def reshape_data(image, perm):
    return (image / 255.0).reshape((-1, 28 * 28))[:, perm]


# ----------------------------
# 4️⃣ Importance Measures
# ----------------------------
def calculate_fim(model, dataloader_fim, perm):
    fim = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    model.eval()

    for sample in dataloader_fim:
        image = reshape_data(sample["image"], perm).to(device)
        label = sample["label"].to(device)

        logits = model(image)
        loss = F.cross_entropy(logits, label)

        model.zero_grad()
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                fim[name] += p.grad.detach() ** 2

    for name in fim:
        fim[name] /= len(dataloader_fim.dataset)

    return fim, deepcopy(model.state_dict())


def compute_gradient_importance(model, dataloader, perm):
    importance = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    model.eval()

    for sample in dataloader:
        image = reshape_data(sample["image"], perm).to(device)
        label = sample["label"].to(device)

        logits = model(image)
        loss = F.cross_entropy(logits, label)

        model.zero_grad()
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                importance[name] += p.grad.abs()

    for name in importance:
        importance[name] /= len(dataloader)

    return importance


def compute_loss_perturbation_importance(model, dataloader, perm, epsilon=1e-3):
    importance = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    model.eval()

    for sample in dataloader:
        image = reshape_data(sample["image"], perm).to(device)
        label = sample["label"].to(device)

        base_loss = F.cross_entropy(model(image), label)

        for name, p in model.named_parameters():
            orig = p.data.clone()

            p.data = orig + epsilon
            loss_pos = F.cross_entropy(model(image), label)

            p.data = orig - epsilon
            loss_neg = F.cross_entropy(model(image), label)

            importance[name] += ((loss_pos + loss_neg - 2 * base_loss).abs() / 2.0)

            p.data = orig

    for name in importance:
        importance[name] /= len(dataloader)

    return importance


def ewc_loss(model, old_model_state, importance_dict):
    loss = 0
    for name, param in model.named_parameters():
        loss += (
            importance_dict[name]
            * (param - old_model_state[name]) ** 2
        ).sum()

    return ewc_lambda * loss


# ----------------------------
# 5️⃣ Training & Evaluation
# ----------------------------
def train_epoch(model, optimizer, dataloader_train, perm,
                fims=None, old_state_dicts=None, use_ewc=False):

    model.train()

    for sample in dataloader_train:
        image = reshape_data(sample["image"], perm).to(device)
        label = sample["label"].to(device)

        logits = model(image)
        loss = F.cross_entropy(logits, label)

        if use_ewc:
            for i in range(len(old_state_dicts)):
                loss += ewc_loss(model, old_state_dicts[i], fims[i])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate_permutations(model, dataloader_test, perms):
    accuracies = []
    model.eval()

    with torch.no_grad():
        for t_idx, perm in enumerate(perms):
            total_correct = 0
            total_samples = 0

            for sample in dataloader_test:
                image = reshape_data(sample["image"], perm).to(device)
                label = sample["label"].to(device)

                logits = model(image)
                pred = logits.argmax(dim=-1)

                total_correct += (pred == label).sum().item()
                total_samples += label.size(0)

            acc = 100.0 * total_correct / total_samples
            accuracies.append(acc)
            print(f"Task {t_idx} accuracy: {acc:.2f}%")

    return accuracies


def train_task(model, optimizer, dataloader_train, dataloader_test,
               perm, eval_perms, fims=None, old_state_dicts=None, use_ewc=False):

    accuracies = torch.zeros((len(eval_perms), epochs))

    for epoch in tqdm.tqdm(range(epochs)):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        epoch_acc = evaluate_permutations(model, dataloader_test, eval_perms)

        for idx, acc in enumerate(epoch_acc):
            accuracies[idx, epoch] = acc / 100

        train_epoch(model, optimizer, dataloader_train,
                    perm, fims, old_state_dicts, use_ewc)

    return accuracies


# ----------------------------
# 6️⃣ Main Execution
# ----------------------------
def main():

    dataloader_train, dataloader_test, dataloader_fim = get_dataloaders(batch_size)

    # Create permutations
    tasks = torch.zeros(n_tasks, 28 * 28).long()
    for t in range(n_tasks):
        tasks[t] = torch.tensor(np.random.permutation(28 * 28))

    # ----------------------------
    # WITHOUT EWC
    # ----------------------------
    model = get_model()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    all_accuracies_no_ewc = []

    for t_id in range(n_tasks):
        print("\n" + "=" * 50)
        print(f"Training Task {t_id} WITHOUT EWC")
        print("=" * 50)

        if t_id == 0:
            run = train_task(model, optimizer, dataloader_train,
                             dataloader_test, tasks[t_id], [tasks[t_id]])
        else:
            run = train_task(model, optimizer, dataloader_train,
                             dataloader_test, tasks[t_id], tasks[:t_id + 1])

        all_accuracies_no_ewc.append(run)

    # ----------------------------
    # WITH EWC
    # ----------------------------
    model = get_model()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    importance_type = "grad"  # "fisher", "grad", "loss_perturb"

    fims = []
    old_models = []
    all_accuracies_ewc = []

    for t_id in range(n_tasks):

        print("\n" + "=" * 50)
        print(f"Training Task {t_id} WITH EWC ({importance_type})")
        print("=" * 50)

        if t_id == 0:
            run = train_task(model, optimizer, dataloader_train,
                             dataloader_test, tasks[t_id], [tasks[t_id]])
        else:
            run = train_task(model, optimizer, dataloader_train,
                             dataloader_test, tasks[t_id], tasks[:t_id + 1],
                             fims, old_models, True)

        all_accuracies_ewc.append(run)

        # Compute importance
        if importance_type == "fisher":
            imp, model_copy = calculate_fim(model, dataloader_fim, tasks[t_id])
        elif importance_type == "grad":
            imp = compute_gradient_importance(model, dataloader_fim, tasks[t_id])
            model_copy = deepcopy(model.state_dict())
        elif importance_type == "loss_perturb":
            imp = compute_loss_perturbation_importance(model, dataloader_fim, tasks[t_id])
            model_copy = deepcopy(model.state_dict())

        fims.append(imp)
        old_models.append(model_copy)

    # ----------------------------
    # Plot
    # ----------------------------
    plt.figure(figsize=(12, 6))

    for idx, acc in enumerate(all_accuracies_no_ewc):
        for t in range(acc.shape[0]):
            plt.plot(acc[t].numpy(), '--')

    for idx, acc in enumerate(all_accuracies_ewc):
        for t in range(acc.shape[0]):
            plt.plot(acc[t].numpy())

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Sequential Learning: With vs Without EWC")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
