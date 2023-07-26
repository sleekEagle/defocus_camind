
from utils_depth import metrics
import utils_depth.logging as logging
import utils
import torch
import math

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']
scale=0.0006

def validate(val_loader, model, criterion_d, device_id, args):

    if device_id == 0:
        depth_loss = logging.AverageMeter()
    model.eval()


    ddp_logger = utils.MetricLogger()
    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    for batch_idx, batch in enumerate(val_loader):
        input_RGB = batch['image'].to(device_id)
        depth_gt = batch['depth'].to(device_id)
        class_id = batch['class_id']
        fdist=batch['fdist'].to(device_id)
        f=batch['f'].to(device_id)
        kcam=(1/(f**2)*scale).float()
        #if(batch_idx>10): break
        with torch.no_grad():
            if args.shift_window_test:
                bs, _, h, w = input_RGB.shape
                assert w > h and bs == 1
                interval_all = w - h
                interval = interval_all // (args.shift_size-1)
                sliding_images = []
                sliding_masks = torch.zeros((bs, 1, h, w), device=input_RGB.device)
                class_ids = []
                for i in range(args.shift_size):
                    sliding_images.append(input_RGB[..., :, i*interval:i*interval+h])
                    sliding_masks[..., :, i*interval:i*interval+h] += 1
                    class_ids.append(class_id)
                input_RGB = torch.cat(sliding_images, dim=0)
                class_ids = torch.cat(class_ids, dim=0)
            if args.flip_test:
                input_RGB = torch.cat((input_RGB, torch.flip(input_RGB, [3])), dim=0)
                class_ids = torch.cat((class_ids, class_ids), dim=0)

            pred_d,_ = model(input_RGB,flag_step2=True,kcam=0)
        if args.flip_test:
            batch_s = pred_d.shape[0]//2
            pred_d = (pred_d[:batch_s] + torch.flip(pred_d[batch_s:], [3]))/2.0
        if args.shift_window_test:
            pred_s = torch.zeros((bs, 1, h, w), device=pred_d.device)
            for i in range(args.shift_size):
                pred_s[..., :, i*interval:i*interval+h] += pred_d[i:i+1]
            pred_d = pred_s/sliding_masks

        pred_d = pred_d.squeeze()
        depth_gt = depth_gt.squeeze()

        loss_d = criterion_d(pred_d.squeeze(), depth_gt)

        ddp_logger.update(loss_d=loss_d.item())

        if device_id == 0:
            depth_loss.update(loss_d.item(), input_RGB.size(0))

        #cropping_img filters out valid depth values. No zero depths after this
        pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)
        computed_result = metrics.eval_depth(pred_crop, gt_crop)


        ddp_logger.update(**computed_result)
        for key in result_metrics.keys():
            result_metrics[key] += computed_result[key]

    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (batch_idx + 1)

    #ddp_logger.synchronize_between_processes()

    for key in result_metrics.keys():
        result_metrics[key] = ddp_logger.meters[key].global_avg

    loss_d = ddp_logger.meters['loss_d'].global_avg
    return result_metrics,loss_d


#provides distance wise error
def validate_dist(val_loader, model, criterion_d, device_id, args,min_dist=0.0,max_dist=10.0,kcam=0):

    if device_id == 0:
        depth_loss = logging.AverageMeter()
    model.eval()


    ddp_logger = utils.MetricLogger()
    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    for batch_idx, batch in enumerate(val_loader):
        input_RGB = batch['image'].to(device_id)
        depth_gt = batch['depth'].to(device_id)
        class_id = batch['class_id']
        #if(batch_idx>10): break
        with torch.no_grad():
            if args.shift_window_test:
                bs, _, h, w = input_RGB.shape
                assert w > h and bs == 1
                interval_all = w - h
                interval = interval_all // (args.shift_size-1)
                sliding_images = []
                sliding_masks = torch.zeros((bs, 1, h, w), device=input_RGB.device)
                class_ids = []
                for i in range(args.shift_size):
                    sliding_images.append(input_RGB[..., :, i*interval:i*interval+h])
                    sliding_masks[..., :, i*interval:i*interval+h] += 1
                    class_ids.append(class_id)
                input_RGB = torch.cat(sliding_images, dim=0)
                class_ids = torch.cat(class_ids, dim=0)
            if args.flip_test:
                input_RGB = torch.cat((input_RGB, torch.flip(input_RGB, [3])), dim=0)
                class_ids = torch.cat((class_ids, class_ids), dim=0)
            pred_d,_ = model(input_RGB,flag_step2=True,kcam=kcam)
        if args.flip_test:
            batch_s = pred_d.shape[0]//2
            pred_d = (pred_d[:batch_s] + torch.flip(pred_d[batch_s:], [3]))/2.0
        if args.shift_window_test:
            pred_s = torch.zeros((bs, 1, h, w), device=pred_d.device)
            for i in range(args.shift_size):
                pred_s[..., :, i*interval:i*interval+h] += pred_d[i:i+1]
            pred_d = pred_s/sliding_masks

        pred_d = (pred_d.squeeze()).squeeze()
        depth_gt = depth_gt.squeeze()

        depth_gt[depth_gt<min_dist]=0.0
        depth_gt[depth_gt>max_dist]=0.0
        #if(torch.sum(depth_gt)==0.0):
        #    print('all zero!')

        loss_d = criterion_d(pred_d, depth_gt)

        ddp_logger.update(loss_d=loss_d.item())

        if device_id == 0:
            depth_loss.update(loss_d.item(), input_RGB.size(0))

        #cropping_img filters out valid depth values. No zero depths after this
        pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)
        computed_result = metrics.eval_depth(pred_crop, gt_crop)
        if math.isnan(computed_result['rmse']):
            continue
        #if rank == 0:
        #    save_path = os.path.join(result_dir, filename)

        #    if save_path.split('.')[-1] == 'jpg':
        #        save_path = save_path.replace('jpg', 'png')

        #    if args.save_result:
        #        if args.dataset == 'kitti':
        #            pred_d_numpy = pred_d.cpu().numpy() * 256.0
        #            cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
        #                        [cv2.IMWRITE_PNG_COMPRESSION, 0])
        #        else:
        #            pred_d_numpy = pred_d.cpu().numpy() * 1000.0
        #            cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
        #                        [cv2.IMWRITE_PNG_COMPRESSION, 0])

        #if rank == 0:
        #    loss_d = depth_loss.avg
        #    if args.pro_bar:
        #        logging.progress_bar(batch_idx, len(val_loader), args.epochs, epoch)

        ddp_logger.update(**computed_result)
        for key in result_metrics.keys():
            result_metrics[key] += computed_result[key]

    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (batch_idx + 1)

    #ddp_logger.synchronize_between_processes()

    for key in result_metrics.keys():
        result_metrics[key] = ddp_logger.meters[key].global_avg

    loss_d = ddp_logger.meters['loss_d'].global_avg
    return result_metrics,loss_d
