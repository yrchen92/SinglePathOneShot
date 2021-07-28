import torch
import numpy as np
import tqdm

assert torch.cuda.is_available()

from utils_glue import compute_metrics, convert_examples_to_features, output_modes, processors

def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func


def get_cand_err(model, cand, args, val_dataset):
    print('starting test....')
    model.eval()
    preds = None
    labels = None
    for batch in val_dataset:
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {'input_ids':      batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] \
                                                and not args.no_segment else None,
                'labels':         batch[3],
                'architecture':  cand}
        with torch.no_grad():
            output = model(**inputs)
        if preds is None:
            preds = output.detach().cpu().numpy()
            labels = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, output.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, inputs['labels'].detach().cpu().numpy(), axis=0)
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(args.task_name, preds, labels)
    # del batch, preds, labels
    print(result)
    return result


def main():
    pass
