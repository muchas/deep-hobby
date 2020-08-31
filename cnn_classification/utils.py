import torch


def find_most_confused(model, test_loader, device, top_n: int = None):
    confused_images = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        probs = torch.softmax(output, dim=1)
        predicted = output.argmax(dim=1).squeeze()
        predicted_probs = torch.take(probs, predicted)

        prob_diff = torch.abs(labels - predicted_probs)
        incorrect = predicted != labels

        probs_inc = [float(x) for x in prob_diff[incorrect]]
        predicted_inc, labels_inc, images_inc = predicted[incorrect], labels[incorrect], images[incorrect]
        result_objs = [{'img': img.cpu(), 'predicted': p.item(), 'expected': e.item()}
                       for img, p, e in zip(images_inc, predicted_inc, labels_inc)]

        confused_images += list(zip(probs_inc, result_objs))
        confused_images = sorted(confused_images, key=lambda x: x[0], reverse=True)

        if top_n is not None:
          confused_images = confused_images[:top_n]

    return confused_images


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
