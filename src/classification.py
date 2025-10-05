import argparse
import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset_stage1 import EEGImageNetDatasetS1
from dataset_stage2 import EEGImageNetDatasetS2
from model.simple_model import SimpleModel
from model.eegnet import EEGNet
from model.mlp import MLP
from model.rgnn import RGNN, get_edge_weight
from src.de_feat_cal import de_feat_cal_S1, de_feat_cal_S2
from utilities import *
from two_way import two_way_identification


def model_init(args, if_simple, num_classes, device):
    if if_simple:
        _model = SimpleModel(args)
    elif args.model.lower() == "eegnet":
        _model = EEGNet(args, num_classes)
    elif args.model.lower() == "mlp":
        _model = MLP(args, num_classes)
    elif args.model.lower() == "rgnn":
        edge_index, edge_weight = get_edge_weight()
        _model = RGNN(device, 62, edge_weight, edge_index, 5, 200, num_classes, 2)
    else:
        raise ValueError(f"Couldn't find the model {args.model}")
    return _model


def model_main(args, model, train_loader, test_loader, criterion, optimizer, num_epochs, device, labels):
    model = model.to(device)
    unique_labels = torch.from_numpy(labels).unique()
    label_mapping = {original_label.item(): new_label for new_label, original_label in enumerate(unique_labels)}
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}
    running_loss = 0.0
    max_acc = 0.0
    max_acc_epoch = -1
    max_twi = 0.0
    max_twi_epoch = -1
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            labels = torch.tensor([label_mapping[label.item()] for label in labels])
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            test_loss = 0
            twi_list = []
            for inputs, labels in test_loader:
                labels = torch.tensor([label_mapping[label.item()] for label in labels])
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, dim=1)
                total += len(labels)
                correct += accuracy_score(labels.cpu(), predicted.cpu(), normalize=False)
                twi = two_way_identification(labels, torch.softmax(outputs, dim=1), label_mapping)
                twi_list.append(twi)
            twi_score = torch.cat(twi_list, dim=0).mean().item()
        acc = correct / total
        if acc > max_acc:
            # print(f"Accuracy on test set: {acc}; Loss on test set: {test_loss / len(test_loader)}")
            max_acc = acc
            max_acc_epoch = epoch
            # torch.save(model.state_dict(), os.path.join(args.output_dir, f"mlp_all.pth"))
        if twi_score > max_twi:
            print(
                f"Accuracy on test set: {acc}; Loss on test set: {test_loss / len(test_loader)}; Two-way identification score: {twi_score}")
            max_twi = twi_score
            max_twi_epoch = epoch
    return max_acc, max_acc_epoch, max_twi, max_twi_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", required=True, help="directory name of EEG-ImageNet dataset path")
    parser.add_argument("-g", "--granularity", required=True, help="choose from coarse, fine0-fine4 and all")
    parser.add_argument("-t", "--task", required=True, help="task to perform, chosen from wt, ct, cp and pt.")
    parser.add_argument("-m", "--model", required=True, help="model")
    parser.add_argument("-b", "--batch_size", default=40, type=int, help="batch size")
    parser.add_argument("-p", "--pretrained_model", help="pretrained model")
    parser.add_argument("-s", "--subject", default=0, type=int, help="subject from 0 to 15")
    parser.add_argument("-o", "--output_dir", required=True, help="directory to save results")
    args = parser.parse_args()
    print(args)

    if args.task == "wt":
        train_set = EEGImageNetDatasetS2(args.dataset_dir, args.subject, args.granularity, 30)
        train_eeg_data = np.stack([i[0].numpy() for i in train_set], axis=0)
        train_de_feat = de_feat_cal_S2(train_eeg_data, args.subject, args.granularity, 30)
        train_set.add_frequency_feat(train_de_feat)
        train_labels = np.array([i[1] for i in train_set])

        test_set = EEGImageNetDatasetS2(args.dataset_dir, args.subject, args.granularity, 20)
        test_eeg_data = np.stack([i[0].numpy() for i in test_set], axis=0)
        test_de_feat = de_feat_cal_S2(test_eeg_data, args.subject, args.granularity, 20)
        test_set.add_frequency_feat(test_de_feat)
        test_labels = np.array([i[1] for i in test_set])
    elif args.task == "ct":
        test_set = EEGImageNetDatasetS2(args.dataset_dir, args.subject, args.granularity, 'all')
        test_eeg_data = np.stack([i[0].numpy() for i in test_set], axis=0)
        test_de_feat = de_feat_cal_S2(test_eeg_data, args.subject, args.granularity, 'all')
        test_set.add_frequency_feat(test_de_feat)
        test_labels = np.array([i[1] for i in test_set])

        train_set = EEGImageNetDatasetS1(args.dataset_dir, same_subject_dict[args.subject], args.granularity)
        train_eeg_data = np.stack([i[0].numpy() for i in train_set], axis=0)
        train_de_feat = de_feat_cal_S1(train_eeg_data, same_subject_dict[args.subject], args.granularity)
        train_set.add_frequency_feat(train_de_feat)
        train_labels = np.array([i[1] for i in train_set])
    elif args.task == "cp":
        test_set = EEGImageNetDatasetS1(args.dataset_dir, same_subject_dict[args.subject], args.granularity)
        test_eeg_data = np.stack([i[0].numpy() for i in test_set], axis=0)
        test_de_feat = de_feat_cal_S1(test_eeg_data, args.subject, args.granularity)
        test_set.add_frequency_feat(test_de_feat)
        test_labels = np.array([i[1] for i in test_set])

        train_set = EEGImageNetDatasetS1(args.dataset_dir, -2, args.granularity)
        train_eeg_data = np.stack([i[0].numpy() for i in train_set], axis=0)
        train_de_feat_list = []
        for subject in range(16):
            if subject in [0, 4, 6, 12, 13]:
                continue
            train_de_feat = de_feat_cal_S1(None, subject, args.granularity)
            train_de_feat_list.append(train_de_feat)
        train_de_feat = np.concatenate(train_de_feat_list, axis=0)
        train_set.add_frequency_feat(train_de_feat)
        train_labels = np.array([i[1] for i in train_set])
    elif args.task == "pt":
        if args.pretrained_model is None:
            print("please specify pretrained model")
            exit(1)
        test_set = EEGImageNetDatasetS2(args.dataset_dir, args.subject, args.granularity, 'all')
        test_eeg_data = np.stack([i[0].numpy() for i in test_set], axis=0)
        test_de_feat = de_feat_cal_S2(test_eeg_data, args.subject, args.granularity, 'all')
        test_set.add_frequency_feat(test_de_feat)
        test_labels = np.array([i[1] for i in test_set])

        train_set = EEGImageNetDatasetS1(args.dataset_dir, same_subject_dict[args.subject], args.granularity)
        train_eeg_data = np.stack([i[0].numpy() for i in train_set], axis=0)
        train_de_feat = de_feat_cal_S1(train_eeg_data, same_subject_dict[args.subject], args.granularity)
        train_set.add_frequency_feat(train_de_feat)
        train_labels = np.array([i[1] for i in train_set])
    else:
        print("unknown task")
        exit(1)

    simple_model_list = ["svm", "rf", "knn", "dt", "ridge"]
    if_simple = args.model.lower() in simple_model_list
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model_init(args, if_simple, len(train_set) // 30, device)
    if args.pretrained_model:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, str(args.pretrained_model))))
    if if_simple:
        unique_labels = torch.from_numpy(train_labels).unique()
        label_mapping = {original_label.item(): new_label for new_label, original_label in enumerate(unique_labels)}
        train_labels = torch.tensor([label_mapping[label.item()] for label in train_labels])
        test_labels = torch.tensor([label_mapping[label.item()] for label in test_labels])
        model.fit(train_de_feat, train_labels)
        y_pred = model.predict(test_de_feat)
        acc = accuracy_score(test_labels, y_pred)
        precision = precision_score(test_labels, y_pred, average="weighted")
        recall = recall_score(test_labels, y_pred, average="weighted")
        f1 = f1_score(test_labels, y_pred, average="weighted")
        probabilities = model.model.predict_proba(test_de_feat)
        twi = two_way_identification(torch.tensor(test_labels), torch.tensor(probabilities))
        with open(os.path.join(args.output_dir, f"{args.model.lower()}.txt"), "a") as f:
            f.write(f"Model: {args.model}\t")
            f.write(f"Accuracy: {acc:.4f}\t")
            f.write(f"Precision: {precision:.4f}\t")
            f.write(f"Recall: {recall:.4f}\t")
            f.write(f"F1 Score: {f1:.4f}\t")
            f.write(f"Two-way Identification Score: {twi.mean().item():.4f}\t")
            f.write("\n")
    else:
        if args.model.lower() == "eegnet":
            train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
            test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-3, momentum=0.9)
            max_acc, max_acc_epoch, max_twi, max_twi_epoch = model_main(
                args, model, train_dataloader, test_dataloader, criterion, optimizer, 1000, device, train_labels
            )
        elif args.model.lower() == "mlp":
            train_set.use_frequency_feat = True
            test_set.use_frequency_feat = True
            train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
            test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-4, momentum=0.9)
            max_acc, max_acc_epoch, max_twi, max_twi_epoch = model_main(
                args, model, train_dataloader, test_dataloader, criterion, optimizer, 1000, device, train_labels
            )
        elif args.model.lower() == "rgnn":
            train_set.use_frequency_feat = True
            test_set.use_frequency_feat = True
            train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
            test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
            max_acc, max_acc_epoch, max_twi, max_twi_epoch = model_main(
                args, model, train_dataloader, test_dataloader, criterion, optimizer, 2000, device, train_labels
            )
        with open(os.path.join(args.output_dir, f"{args.model.lower()}.txt"), "a") as f:
            f.write(f"{args.granularity}\t{args.subject}\t{max_acc}\t{max_acc_epoch}\t{max_twi}\t{max_twi_epoch}\n")
