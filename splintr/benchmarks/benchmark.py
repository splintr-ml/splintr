import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import splintr.benchmarks.data_loader as dl

""" CIFAR100 Benchmark """

def cifar100(model, wrapper = None, batch_size = 128, target_acc = 1, target_top10 = 1, evaluate = True, verbose = 3):

    train, test = dl.cifar100()
    train, test = list(train), list(test)

    if verbose:
        print('--- Data Loaded ---')

    device = torch.device('cpu')

    if torch.cuda.is_available() and not wrapper:
        device = torch.device('cuda')

    if wrapper:
        model = wrapper(model)
    else:
        model = model.to(device)

    train_images = torch.from_numpy(np.array([np.moveaxis(i[0], 2, 0) for i in train])).float()
    test_images = torch.from_numpy(np.array([np.moveaxis(i[0], 2, 0) for i in test])).float()

    if not wrapper:
        train_images = train_images.to(device)
        test_images = test_images.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.01)

    tot_time = 0
    num_mb_epoch = int((50000 / batch_size)) + 1
    mb_left = 30 * num_mb_epoch
    for epoch in range(30):
        avg_batch_time = 0
        loss_over_minibatch = 0
        prev_time = time.time()

        for mb in range(num_mb_epoch):
            inputs = train_images[mb * batch_size:(mb + 1) * batch_size]
            labels = torch.from_numpy(np.array([i[1] for i in train[mb * batch_size:(mb + 1) * batch_size]])).long()

            if not wrapper:
                labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_func(outputs, labels.to(outputs.device))
            loss.backward()
            optimizer.step()

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(torch.device('cuda', i))

            tot_time += time.time() - prev_time
            avg_batch_time += time.time() - prev_time
            loss_over_minibatch += loss.item()
            prev_time = time.time()
            mb_left -= 1

            if (not wrapper) and (torch.cuda.is_available()):
                del inputs
                del labels
                del outputs
                torch.cuda.empty_cache()

            if ((mb % 50) == 49) and (verbose == 3) and evaluate:
                print('Epoch ' + str(epoch + 1) + ' Loss: ' + str(loss_over_minibatch / 50))
                print('Time Per Batch: ' + str(avg_batch_time / 50) + 's Time Left: ' + str((avg_batch_time / 50) * mb_left) + 's')

                inputs = test_images
                labels = torch.from_numpy(np.array([i[1] for i in test])).long()

                if not wrapper:
                    labels = labels.to(device)

                model.eval()
                with torch.no_grad():
                    outputs = model(inputs)
                model.train()

                _, predicted = torch.max(outputs, 1)
                _, top10 = torch.topk(outputs, 10, dim = 1)

                predicted = predicted.to(labels.device)
                top10 = top10.to(labels.device)

                corr = 0
                top10_corr = 0

                for i in range(len(predicted)):
                    if predicted[i].item() == labels[i]:
                        corr += 1
                    if labels[i] in top10[i]:
                        top10_corr += 1

                print('Acc: ' + str(corr / len(predicted)))
                print('Top-10: ' + str(top10_corr / len(predicted)))

                if ((corr / len(predicted)) >= target_acc) and ((top10_corr / len(predicted)) >= target_top10):
                    print("Total Time: " + str(tot_time) + "s")
                    return

                if not wrapper and torch.cuda.is_available():
                    del inputs
                    del labels
                    del outputs
                    del predicted
                    del top10
                    torch.cuda.empty_cache()

                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.synchronize(torch.device('cuda', i))

                avg_batch_time = 0
                loss_over_minibatch = 0
                prev_time = time.time()

    if (verbose >= 2) and evaluate:
        inputs = test_images
        labels = torch.from_numpy(np.array([i[1] for i in test])).long()

        if not wrapper:
            labels = labels.to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        model.train()

        _, predicted = torch.max(outputs, 1)
        _, top10 = torch.topk(outputs, 10, dim = 1)

        predicted = predicted.to(labels.device)
        top10 = top10.to(labels.device)

        corr = 0
        top10_corr = 0

        for i in range(len(predicted)):
            if predicted[i].item() == labels[i]:
                corr += 1
            if labels[i] in top10[i]:
                top10_corr += 1

        print('Acc: ' + str(corr / len(predicted)))
        print('Top-10: ' + str(top10_corr / len(predicted)))

        if ((corr / len(predicted)) >= target_acc) and ((top10_corr / len(predicted)) >= target_top10):
            print("Total Time: " + str(tot_time) + "s")
            return

        if not wrapper and torch.cuda.is_available():
            del inputs
            del labels
            del outputs
            del predicted
            del top10
            torch.cuda.empty_cache()

    del model

    if verbose:
        print("Total Time: " + str(tot_time) + "s")
