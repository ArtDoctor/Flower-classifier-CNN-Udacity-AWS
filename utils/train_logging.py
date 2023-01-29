import torch


def calc_accuracy(network, dataloader, using_cuda):
    network.eval()
    total_correct = 0
    total_instances = 0

    for images, labels in dataloader:
        if using_cuda:
            images, labels = images.cuda(), labels.cuda()

        classifications = torch.argmax(network(images), dim=1)

        correct_predictions = sum(classifications == labels).item()

        total_correct += correct_predictions
        total_instances += len(images)

    return round(total_correct / total_instances, 3)


def calc_loss(network, dataloader, loss_function, using_cuda):
    loss = 0
    network.eval()
    for images, labels in dataloader:
        if using_cuda:
            images, labels = images.cuda(), labels.cuda()

        classifications = network(images)

        loss += loss_function(classifications, labels).item()

    return round(loss / 160, 3)
