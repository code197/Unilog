import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch.optim as optim
from gensim.models import Word2Vec
from tqdm import tqdm, trange
import time, os, json, pickle, argparse, random, torch, sys, logging
import numpy as np
import pandas as pd
from datetime import datetime
from utils import log_dataset_processors as processors, compute_metrics, criterion_metrics
from transformers import get_linear_schedule_with_warmup
from model import unilog

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

result_save = []
loss_save = []

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def create_model(args):
    return unilog(args.vector_size, args.max_seq_len, args.textcnn_out_channels, args.bilstm_width, args.bilstm_depth, args.dropout, args.device)

def train(args, model, train_dataset):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size) 
    
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // len(train_dataloader) + 1
    else:
        t_total = len(train_dataloader) * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer_class = eval('optim.%s' % args.optimizer)
    if "adam" in args.optimizer.lower():
        optimizer = optimizer_class(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    elif "sgd" in args.optimizer.lower():
        optimizer = optimizer_class(optimizer_grouped_parameters, lr=args.learning_rate, momentum=args.sgd_momentum)
    else:
        optimizer = optimizer_class(optimizer_grouped_parameters, lr=args.learning_rate)
        
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    ) 
    
    if args.loss_function == "CrossEntropyLoss":
        criterion = torch.nn.CrossEntropyLoss()

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size = %d",
        args.train_batch_size
    )
    logger.info("  Total optimization steps = %d", t_total)
    print("***** Running training *****")
    print("  Num examples = %d" % (len(train_dataset)))
    print("  Num Epochs = %d" % (args.num_train_epochs))
    print("  Instantaneous batch size per GPU = %d" % (args.per_gpu_train_batch_size))
    print(
        "  Total train batch size = %d" %
        (args.train_batch_size)
    )
    print("  Total optimization steps = %d" % (t_total))

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if (not args.overwrite_output_dir) and os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)) > 0:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = max([int(d.split("/")[-1].split("-")[-1]) for d in os.listdir(args.output_dir)])
        epochs_trained = global_step // len(train_dataloader)
        steps_trained_in_current_epoch = global_step % len(train_dataloader)

        load_dir = os.path.join(args.output_dir, "checkpoint-last-%d" % (global_step))
        model = torch.load(os.path.join(load_dir, "checkpoint-last"))
        model.to(args.device)
        # Load in last optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.output_dir, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.output_dir, "scheduler.pt")))

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    # model.zero_grad()
    optimizer.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch"
    ) 
    results = {}
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            # preprocess input data
            inputs = {
                "inputs_for_textcnn": batch[1], 
                "inputs_for_bilstm": torch.cat([batch[2][i, :, :].unsqueeze(1) for i in range(batch[2].size(0))], dim=1)
                }
            labels = batch[3]
            
            # forward
            outputs = model(inputs)

            # backward
            ### CrossEntropyLoss()的输入格式
            ### 第1个参数 Input: (N,C) C 是类别的数量
            ### 第2个参数 Target: (N) N是mini-batch的大小，0 <= targets[i] <= C-1
            loss = criterion(outputs, labels) ### labels.long()
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            tr_loss += loss.item()

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            # model.zero_grad()
            optimizer.zero_grad()
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logs = {}
                if args.evaluate_during_training:  
                    # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(args, model, results, "dev")
                    for key, value in results.items():
                        eval_key = "eval_{}".format(key)
                        logs[eval_key] = value

                loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                learning_rate_scalar = scheduler.get_lr()[0]
                logs["learning_rate"] = learning_rate_scalar
                logs["loss"] = loss_scalar
                logging_loss = tr_loss

                # for key, value in logs.items():
                #     tb_writer.add_scalar(key, value, global_step)
                logger.info(json.dumps({**logs, **{"step": global_step}}))
                print(json.dumps({**logs, **{"step": global_step}}))
                
                result_save.append(logs["eval_f1"])
                loss_save.append(logs["loss"])

            if args.save_steps > 0 and global_step % args.save_steps == 0 or global_step == args.max_steps - 1:
                # Save model checkpoint
                save_dir = os.path.join(args.output_dir, "checkpoint-last-%d" % (global_step))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                torch.save(model_to_save, os.path.join(save_dir, "checkpoint-last"))

                torch.save(args, os.path.join(save_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", save_dir)
                print("Saving model checkpoint to %s" % (save_dir))

                torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", save_dir)
                print("Saving optimizer and scheduler states to %s" % (save_dir))

                with open(os.path.join(save_dir, "eval_results.txt"), "w") as writer:
                    for key in sorted(results.keys()):
                        writer.write("%s = %s\n" % (key, str(results[key])))

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step

def evaluate(args, model, results, aim):
    ### results为字典类型
    eval_dataset = load_and_cache_examples(args, aim)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    if args.loss_function == "CrossEntropyLoss":
        criterion = torch.nn.CrossEntropyLoss()

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    print("***** Running evaluation *****")
    print("  Num examples = %d" % (len(eval_dataset)))
    print("  Batch size = %d" % (args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        ids = batch[0]

        with torch.no_grad():
            # preprocess input data
            inputs = {
                "inputs_for_textcnn": batch[1], 
                "inputs_for_bilstm": torch.cat([batch[2][i, :, :].unsqueeze(1) for i in range(batch[2].size(0))], dim=1)
                }
            labels = batch[3]

            outputs = model(inputs)

            # backward
            ### CrossEntropyLoss()的输入格式
            ### 第1个参数 Input: (N,C) C 是类别的数量
            ### 第2个参数 Target: (N) N是mini-batch的大小，0 <= targets[i] <= C-1
            loss = criterion(outputs, labels) ### labels.long()
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            eval_loss += loss.item()

        nb_eval_steps += 1
        if preds is None:
            example_ids = ids.detach().cpu().numpy()
            preds = outputs.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            example_ids = np.append(example_ids, ids.detach().cpu().numpy(), axis=0)
            preds = np.append(preds, outputs.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    results_last = compute_metrics(preds, out_label_ids)
    results.update(results_last)
    criterion_val = criterion_metrics(results_last)
    if "best_criterion" not in results or criterion_val > results['best_criterion']:
        results["best_criterion"] = criterion_val
        
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        torch.save(model_to_save, os.path.join(args.output_dir, "checkpoint-best"))

    logger.info("***** Eval results *****")
    print("***** Eval results *****")
    for key in sorted(results_last.keys()):
        logger.info("  %s = %s", key, str(results_last[key]))
        print("  %s = %s" % (key, str(results_last[key])))

    if aim == "test" or aim == "all":
        example_ids_correct = example_ids[np.where(preds == out_label_ids)[0]].tolist()
        w = open(os.path.join(args.output_dir, "example_ids_test_correct"), "w")
        for i in range(len(example_ids_correct)):
            w.write("%d\n" % (example_ids_correct[i]))
        w.close()
        example_ids_error = example_ids[np.where(preds != out_label_ids)[0]].tolist()
        w = open(os.path.join(args.output_dir, "example_ids_test_error"), "w")
        for i in range(len(example_ids_error)):
            w.write("%d\n" % (example_ids_error[i]))
        w.close()

    return results

### aim = "train", "dev", "test"
def load_and_cache_examples(args, aim="train"):
    processor = processors[args.dataset_name]()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}_{}_{}_{}".format(
            aim,
            str(args.train_percent),
            str(args.dataset_name),
            str(args.positive_rate),
            str(args.window_size),
            str(args.vector_size),
            str(args.max_seq_len),
        ),
    )

    logger.info("Loading features from cached file %s", cached_features_file)
    print("Loading features from cached file %s" % (cached_features_file))
    features = torch.load(cached_features_file)

    # Convert to Tensors and build dataset
    all_ids = torch.tensor([f["ids"] for f in features], dtype=torch.long)
    all_inputs_for_textcnn = torch.tensor([f["inputs_for_textcnn"] for f in features], dtype=torch.float)
    all_inputs_for_bilstm = torch.tensor([f["inputs_for_bilstm"] for f in features], dtype=torch.float)
    all_labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)

    dataset = TensorDataset(all_ids, all_inputs_for_textcnn, all_inputs_for_bilstm, all_labels)
    return dataset

def main():
    parser = argparse.ArgumentParser()
    # EXPERIMENT SETTINGS
    parser.add_argument('--gpu', type=str, default='0', help='GPU No.')
    parser.add_argument('--no_cuda', action='store_true', help='Avoid using CUDA when available')
    parser.add_argument('--n_worker', type=int, default=16, help='The number of CPUs.')
    parser.add_argument('--data_dir', 
        default=None,
        type=str, 
        required=True,
        help='Data direction.')
    parser.add_argument('--dataset_name', 
        default=None,
        type=str, 
        required=True,
        help='The name of log data set.')
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",)
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--test_all", action="store_true", help="Whether to test all the data.")
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model",)
    parser.add_argument("--positive_rate", type=float, help="Set the rate of positive examples",)
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    # MODEL HYPERPARAMETERS
    parser.add_argument('--window_size', type=int, default=5, help='Size of time window')
    parser.add_argument('--vector_size', type=int, default=5, help='Size of word vector')
    parser.add_argument('--max_seq_len', type=int, default=20, help='Maximal length of sequence')
    parser.add_argument('--textcnn_out_channels', type=int, default=10, help='Number of TextCNN out channels')
    parser.add_argument('--bilstm_width', type=int, default=200, help='Width of BiLSTM hidden layers')
    parser.add_argument('--bilstm_depth', type=int, default=8, help='Number of BiLSTM hidden layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--learning_rate', type=float, default=4.7041e-05, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.033218, help='Weight decay.')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Epsilon for Adam optimizer.')
    parser.add_argument('--sgd_momentum', type=float, default=0.9, help='Momentum for SGD optimizer.')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='Max gradient norm.')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
    parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss', help='Loss Function')
    # STOPPING CRITERIA
    parser.add_argument('--train_percent', type=float, default=0.8, help='Percentage of training set.')
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument("--per_gpu_train_batch_size", default=200, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=200, type=int, help="Batch size per GPU/CPU for evaluation.")
    #parser.add_argument('--consecutive', type=int, default= 2000, help='Consecutive 100% training accuracy to stop')
    #parser.add_argument('--early_stopping', type=int, default= 100, help='Early Stopping')
    #parser.add_argument('--epochs_after_peak', type=int, default=200, help='Number of More Epochs Needed after 100% Training Accuracy Happens')
    # MISC
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--amp', type=int, default=2, help='1, 2 and 3 for NVIDIA apex amp optimization O1, O2 and O3, 0 for off')
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument('--expname', type=str, default="", required=True)

    args = parser.parse_args()

    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUC_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "Device: %s, n_gpu: %s, 16-bits training: %s",
        args.device,
        args.n_gpu,
        args.fp16,
    )

    set_seed(args) # Added here for reproductibility

    # Prepare log dataset
    args.dataset_name = args.dataset_name.lower()
    if args.dataset_name not in processors:
        raise ValueError("Task not found: %s" % (args.dataset_name))
    processor = processors[args.dataset_name]()

    model = create_model(args)
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    print("Training/evaluation parameters %s" % (args))

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, aim="train")
        global_step, tr_loss = train(args, model, train_dataset)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        print(" global_step = %s, average loss = %s" % (global_step, tr_loss))

        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        print("Saving model checkpoint to %s" % (args.output_dir))

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluate the best checkpoint
    results = {}
    if args.do_eval:
        checkpoints = [os.path.join(args.output_dir, "checkpoint-best")]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        print("Evaluate the following checkpoints: %s" % (checkpoints))
        for checkpoint in checkpoints:
            model = torch.load(checkpoint)
            model.to(args.device)
            results = evaluate(args, model, results, "all" if args.test_all else "test")
            result_final = dict((k + "_{}".format("best"), v) for k, v in results.items())
            logger.info(result_final)
            print(result_final)
    """
    # Evaluate a checkpoint
    results = {}
    if args.do_eval:
        checkpoints = [os.path.join(args.output_dir, "checkpoint-last-3200/checkpoint-last")]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        print("Evaluate the following checkpoints: %s" % (checkpoints))
        for checkpoint in checkpoints:
            model = torch.load(checkpoint)
            model.to(args.device)
            results = evaluate(args, model, results, "all" if args.test_all else "test")
            result_final = dict((k + "_{}".format("best"), v) for k, v in results.items())
            logger.info(result_final)
            print(result_final)
    """
    # Save result and loss （当前目录是运行程序时所在的目录）
    f = open(os.path.join(args.output_dir, "%s_result.json" % (args.expname)), "w")
    json.dump(obj=result_save, fp=f, indent=4)
    f.close()
    f = open(os.path.join(args.output_dir, "%s_loss.json" % (args.expname)), "w")
    json.dump(obj=loss_save, fp=f, indent=4)
    f.close()

if __name__ == "__main__":
    main()