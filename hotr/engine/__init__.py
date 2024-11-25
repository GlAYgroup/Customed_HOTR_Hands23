from .evaluator_vcoco import vcoco_evaluate, vcoco_accumulate, vcoco_visualizer
from .evaluator_hico import hico_evaluate
from .evaluator_doh import doh_evaluate, doh_visualizer, doh_accumulate



def hoi_evaluator(args, model, criterion, postprocessors, data_loader, device, thr=0):
    if args.dataset_file == 'vcoco':
        return vcoco_evaluate(model, criterion, postprocessors, data_loader, device, args.output_dir, thr)
    elif args.dataset_file == 'hico-det':
        return hico_evaluate(model, postprocessors, data_loader, device, thr)
    elif args.dataset_file == 'doh':
        return doh_evaluate(model, criterion, postprocessors, data_loader, device, args.output_dir, thr, args)
    elif args.dataset_file == 'hands23':
        return doh_evaluate(model, criterion, postprocessors, data_loader, device, args.output_dir, thr, args)
    else: raise NotImplementedError

def hoi_visualizer(args, total_res ,dataset):
    if args.dataset_file == 'vcoco':
        return vcoco_visualizer(args, total_res, dataset)
    # elif args.dataset_file == 'hico-det':
    #     return hico_visualizer(model, postprocessors, data_loader, device, thr)
    elif args.dataset_file == 'doh':
        return doh_visualizer(args, total_res, dataset)
    elif args.dataset_file == 'hands23':
        return doh_visualizer(args, total_res, dataset)
    else: raise NotImplementedError


def hoi_accumulator(args, total_res, print_results=False, wandb=False):
    if args.dataset_file == 'vcoco':
        return vcoco_accumulate(total_res, args, print_results, wandb)
    elif args.dataset_file == 'doh':
        return doh_accumulate(total_res, args, print_results, wandb)
    elif args.dataset_file == 'hands23':
        return doh_accumulate(total_res, args, print_results, wandb)
    else: raise NotImplementedError