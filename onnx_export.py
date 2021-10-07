"""Convert pytorch lightning model to ONNX format to be able to use it from IntelliJ PLugin"""
import argparse
import torch

from data_processing import (
    read_comment_location_file,
    create_vocabularies,
    read_w2v_models,
    DataModule,
)
from model import HierarchicalLSTM


def main():
    parser = argparse.ArgumentParser(description="Train a PyTorch Lightning model")

    parser.add_argument(
        "--data_train_path", default="data/split30_word_data/train/data.txt", help="",
    )
    parser.add_argument(
        "--data_test_path", default="data/split30_word_data/test/data.txt", help="",
    )
    parser.add_argument(
        "--data_validation_path",
        default="data/split30_word_data/valid/data.txt",
        help="",
    )

    parser.add_argument(
        "--check_point_dir",
        default="model_save/",
        help="Directory for saving model during training",
    )

    parser.add_argument("--embed_size", default=300, type=int, help="")
    parser.add_argument("--max_blocks", default=30, type=int, help="")
    parser.add_argument("--max_length_block", default=30, type=int, help="")
    parser.add_argument("--max_length_sentence", default=30, type=int, help="")
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
        default="model-epoch=00-val_loss=0.57.ckpt",
        help="Path for model to predict",
    )
    parser.add_argument("--word_vectors_file", default="word_vectors.txt")
    parser.add_argument("--word_to_indexes", default="all_word_w2i.txt")
    parser.add_argument("--onnx_model_directory", default="model.onnx")

    args = parser.parse_args()

    file_sequences, all_words = read_comment_location_file(
        args.data_train_path, args.max_length_sentence, True,
    )
    file_sequences_val, _ = read_comment_location_file(
        args.data_validation_path, args.max_length_sentence, True,
    )

    file_sequences_test, _ = read_comment_location_file(
        args.data_test_path, args.max_length_sentence, True,
    )

    (wordv, all_wordv, all_word_w2i, all_word_i2w) = create_vocabularies(
        all_words, args.vocab_size
    )
    w2v_models, w2v_dims = read_w2v_models(args.code_embeddings)
    data = DataModule(
        args,
        w2v_models,
        w2v_dims,
        wordv,
        all_wordv,
        all_word_w2i,
        all_word_i2w,
        file_sequences,
        file_sequences_val,
        file_sequences_test,
    )

    args.vectors_store = data.pretrainedvecs

    print(f"Loading the model from checkpoint at '{args.model_for_prediction}'")
    classifier = HierarchicalLSTM(args)
    model = classifier.load_from_checkpoint(
        args.model_for_prediction, kwargs=dict(args=args)
    )
    print(model)

    # Input to the model
    input_x, input_x_weights, targets, targets_weights = next(iter(data.get_train()))
    input = [
        (torch.tensor(input_x)).unsqueeze(0),
        (torch.tensor(input_x_weights)).unsqueeze(0),
        (torch.tensor(targets)).unsqueeze(0),
        (torch.tensor(targets_weights)).unsqueeze(0),
    ]

    print(f"Saving the model in ONNX format to '{args.onnx_model_directory}'")
    torch.onnx.export(
        model,
        input,
        args.onnx_model_directory,
        input_names=["input", "input_weights"],
        opset_version=11,
    )

    print("Done!")


if __name__ == "__main__":
    main()
