import numpy as np
import torch.nn.functional as F


def testing(model, x_cv, le, valid_loader, loss_fn, opt, test_y, accuracy_list):
    model.eval()
    avg_val_loss = 0.
    val_preds = np.zeros((len(x_cv), len(le.classes_)))
    valid_loss = []

    for i, (x_batch, y_batch) in enumerate(valid_loader):
        y_pred = model(x_batch).detach()
        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
        # keep/store predictions
        val_preds[i * opt.batch_size:(i + 1) * opt.batch_size] = F.softmax(y_pred, dim=1).cpu().numpy()

    # Check Accuracy
    val_accuracy = sum(val_preds.argmax(axis=1) == test_y) / len(test_y)
    valid_loss.append(avg_val_loss)
    accuracy_list.append(val_accuracy)
    print('\t val_loss={:.4f}  \t val_acc={:.4f} '.format(avg_val_loss, val_accuracy))
    return accuracy_list
