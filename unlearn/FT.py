import time
import torch
import utils
from .impl import iterative_unlearn
import sys
sys.path.append(".")
from imagenet import get_x_y_from_data_dict

def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)

def l2_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=2)

def FT_iter(data_loaders, model, criterion, optimizer, epoch, args, with_l1=False):
    print("FT_iter")
    with torch.autograd.set_detect_anomaly(True):
        train_loader = data_loaders["retain"]

        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()

        # switch to train mode
        model.train()

        start = time.time()
        if args.imagenet_arch:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            for i,data in enumerate(train_loader):
                image, target = get_x_y_from_data_dict(data, device)
                if epoch < args.warmup:
                    utils.warmup_lr(epoch, i+1, optimizer,
                                    one_epoch_step=len(train_loader), args=args)

                # compute output
                output_clean = model(image)
                if epoch < args.unlearn_epochs-args.no_l1_epochs:
                    current_alpha = args.alpha * (1 - epoch / (args.unlearn_epochs-args.no_l1_epochs))
                else:
                    current_alpha = 0
                loss = criterion(output_clean, target)
                if with_l1:
                    loss = loss + current_alpha * l1_regularization(model)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                output = output_clean.float()
                loss = loss.float()
                # measure accuracy and record loss
                prec1 = utils.accuracy(output.data, target)[0]

                losses.update(loss.item(), image.size(0))
                top1.update(prec1.item(), image.size(0))

                if (i + 1) % args.print_freq == 0:
                    end = time.time()
                    print('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Time {3:.2f}'.format(
                            epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
                    start = time.time()      
        else:
            for i, (image, target) in enumerate(train_loader):
                if epoch < args.warmup:
                    utils.warmup_lr(epoch, i+1, optimizer,
                                    one_epoch_step=len(train_loader), args=args)

                image = image.cuda()
                target = target.cuda()
                if epoch < args.unlearn_epochs-args.no_l1_epochs:
                    current_alpha = args.alpha * (1 - epoch / (args.unlearn_epochs-args.no_l1_epochs))
                    # current_alpha = args.alpha * (epoch / (args.unlearn_epochs-args.no_l1_epochs))
                elif args.unlearn_epochs-args.no_l1_epochs == 0:
                    current_alpha = args.alpha
                else:
                    current_alpha = 0 
                              
                # compute output
                if args.hf_vit=="YES":
                    output_clean = model(image).logits
                else:
                    output_clean = model(image)
    
                loss = criterion(output_clean, target)
                if with_l1:
                    loss += current_alpha * l1_regularization(model)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                output = output_clean.float()
                loss = loss.float()
                # measure accuracy and record loss
                prec1 = utils.accuracy(output.data, target)[0]

                losses.update(loss.item(), image.size(0))
                top1.update(prec1.item(), image.size(0))

                if (i + 1) % args.print_freq == 0:
                    end = time.time()
                    print('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Time {3:.2f}'.format(
                            epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
                    start = time.time()

        print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


@iterative_unlearn
def FT(data_loaders, model, criterion, optimizer, epoch, args):
    return FT_iter(data_loaders, model, criterion, optimizer, epoch, args)


@iterative_unlearn
def FT_l1(data_loaders, model, criterion, optimizer, epoch, args):
    return FT_iter(data_loaders, model, criterion, optimizer, epoch, args, with_l1=True)
