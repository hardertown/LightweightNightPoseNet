for imgs, target_x, target_y, target_weight in val_bar:
    imgs, target_x, target_y, target_weight = imgs.to(device), target_x.to(device), target_y.to(
        device), target_weight.to(device)
    pred_x, pred_y = Net(imgs)
    loss = Loss_fun(pred_x, pred_y, target_x, target_y, target_weight)