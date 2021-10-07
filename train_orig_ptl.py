import argparse

from einops import repeat
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import HierarchicalLSTM
from data_processing import (
    read_comment_location_file,
    create_vocabularies,
    read_w2v_models,
    DataModule,
)

parser = argparse.ArgumentParser(description="Train a PyTorch Lightning model")

parser.add_argument(
    "--data_train_path", default="data/split30_word_data/train/data.txt", help="",
)
parser.add_argument(
    "--data_test_path", default="data/split30_word_data/test/data.txt", help="",
)
parser.add_argument(
    "--data_validation_path", default="data/split30_word_data/valid/data.txt", help="",
)

parser.add_argument(
    "--check_point_dir",
    default="model_save/",
    help="Directory for saving model during training",
)

parser.add_argument("--embed_size", default=300, type=int, help="")
parser.add_argument("--max_blocks", default=10, type=int, help="")
parser.add_argument("--max_length_block", default=10, type=int, help="")
parser.add_argument("--max_length_sentence", default=10, type=int, help="")
parser.add_argument("--vocab_size", default=5000, type=int, help="")
parser.add_argument("--embedding_type", default="avg_wembed", help="")
parser.add_argument(
    "--code_embeddings",
    default="data/split30_word_data/code-big-vectors-negative300.bin",
    help="",
)

# Model related arguments #
parser.add_argument("--use_gpu", default=0, type=int, help="1 for GPU usage")
parser.add_argument("--lstm_hidden_size", default=800, type=int, help="")
parser.add_argument("--lstm_num_layers", default=1, help="")
parser.add_argument(
    "--batch_size", default=64, type=int, help="Batch size for training"
)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--max_epochs", default=1, type=int)
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--val_batch_size", default=64, type=int)
parser.add_argument("--test_batch_size", default=1, type=int)
parser.add_argument("--mode", default="train", help="train or predict")
parser.add_argument(
    "--model_for_prediction",
    default="model_save/model-epoch=00-val_loss=0.50.ckpt",
    help="Path for model to predict",
)
parser.add_argument("--word_vectors_file", default="word_vectors.txt")
parser.add_argument("--word_to_indexes", default="all_word_w2i.txt")

args = parser.parse_args()

file_seqs, all_words = read_comment_location_file(
    args.data_train_path, args.max_length_sentence, True,
)
file_seqs_val, _ = read_comment_location_file(
    args.data_validation_path, args.max_length_sentence, True,
)

file_seqs_test, _ = read_comment_location_file(
    args.data_test_path, args.max_length_sentence, True,
)

(wordv, all_wordv, all_word_w2i, all_word_i2w) = create_vocabularies(
    all_words, args.vocab_size
)

input_file = open(args.word_vectors_file, "w")
for element in wordv:
    input_file.write(element + "\n")
input_file.close()

input_file = open(args.word_to_indexes, "w")
for element in all_word_w2i:
    input_file.write(element + " " + str(all_word_w2i[element]) + "\n")
input_file.close()

w2v_models, w2v_dims = read_w2v_models(args.code_embeddings)

data_module = DataModule(
    args,
    w2v_models,
    w2v_dims,
    wordv,
    all_wordv,
    all_word_w2i,
    all_word_i2w,
    file_seqs,
    file_seqs_val,
    file_seqs_test,
)

args.vectors_store = data_module.pretrainedvecs

if args.mode == "train":
    classifier = HierarchicalLSTM(args)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.check_point_dir,
        every_n_val_epochs=1,
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
        save_weights_only=False,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.use_gpu,
        logger=TensorBoardLogger("logs"),
        gradient_clip_val=50,
        gradient_clip_algorithm="norm",
        track_grad_norm=2,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        classifier, data_module.train_dataloader(), data_module.val_dataloader()
    )

elif args.mode == "predict":
    if args.model_for_prediction == "none":
        print("Model is not loaded.")
        exit()

    classifier = HierarchicalLSTM(args)
    model = classifier.load_from_checkpoint(
        args.model_for_prediction, kwargs=dict(args=args)
    )
    print(model)
    trainer = pl.Trainer(gpus=args.use_gpu)
    test_loader = data_module.test_dataloader()
    result = trainer.test(model=model, test_dataloaders=test_loader)
    print(result)
