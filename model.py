from einops import repeat
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from evaluation import calculate_metrics

# To see Tensorboard loggers after training, you can go http://localhost:6006/

class HierarchicalLSTM(pl.LightningModule):
    """
    It is a Hierarchical Sequence Model based on paper https://homes.cs.washington.edu/~mernst/pubs/predict-comments-icse2020.pdf
    Two levels of RNN:
    First level: Representations of lines of code based on previous lines in file
    Second level: Representations of code blocks based on previous code blocks in file
    Output of model: Binary classification: 1:commentable block, 0:not commentable block
    """

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.number_of_blocks = args.max_blocks
        self.mode = args.mode
        self.number_of_sentences = args.max_length_block
        self.number_of_words = args.max_length_sentence
        self.vocab_size = args.vocab_size
        self.embed_type = args.embedding_type
        self.embed_size = args.embed_size
        self.lstm_hidden_size = args.lstm_hidden_size
        self.lstm_num_layers = args.lstm_num_layers
        self.vectors_store = args.vectors_store
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.val_batch_size = args.val_batch_size
        self.learning_rate = args.learning_rate
        self.embedding = torch.nn.Embedding.from_pretrained(
            torch.Tensor(self.vectors_store), freeze=False
        )
        self.rnn = torch.nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
            dropout=0.1,
            batch_first=True,
        )
        self.rnn2 = torch.nn.LSTM(
            input_size=self.lstm_hidden_size,
            hidden_size=self.lstm_hidden_size,
            batch_first=True,
            dropout=0.1,
            num_layers=1,
        )
        self.linear = torch.nn.Linear(self.lstm_hidden_size, 1)

    def forward(self, batch):
        data, data_wts, targets, target_weights = batch
        embedding_out = self.embedding(data)
        batch_size = data.shape[0]
        data_wts_reshaped = torch.reshape(
            data_wts,
            [
                batch_size,
                self.number_of_blocks,
                self.number_of_sentences,
                self.number_of_words,
                1,
            ],
        )
        repeated_tensor = repeat(
            data_wts_reshaped, "h w d e v ->h w d e (repeat v)", repeat=300
        )
        multiplied = torch.multiply(repeated_tensor, embedding_out)
        # average of words for each sentence
        input_embeddings = torch.reshape(
            torch.mean(multiplied, 3),
            [
                batch_size * self.number_of_blocks,
                self.number_of_sentences,
                self.embed_size,
            ],
        )
        sentence_output, _ = self.rnn(input_embeddings)
        block_representation = torch.reshape(
            torch.max(sentence_output, 1).values,
            [batch_size, self.number_of_blocks, self.lstm_hidden_size],
        )
        block_output, _ = self.rnn2(block_representation)
        output = torch.reshape(block_output, [batch_size * self.number_of_blocks, -1])
        output = torch.reshape(self.linear(output), [batch_size, -1, 1])
        predictions = torch.sigmoid(output)
        return predictions

    def training_step(self, batch, batch_idx):
        input, input_weights, targets, targets_weights = batch
        pred = self(batch)
        result = {}
        loss = self.myloss(pred, targets[:, :, None].float(), targets_weights)
        result["loss"] = loss
        accuracy, precision, recall, fscore, result_string, = calculate_metrics(
            pred, targets, targets_weights
        )
        result["train_precision"] = torch.tensor(precision)
        result["train_recall"] = torch.tensor(recall)
        return result

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([(x["loss"]) for x in outputs]).mean()
        avg_precision = torch.stack([(x["train_precision"]) for x in outputs]).mean()
        avg_recall = torch.stack([(x["train_recall"]) for x in outputs]).mean()

        # For tensorboard logging
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar(
            "Precision/Train", avg_precision, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "Recall/Train", avg_recall, self.current_epoch
        )

        return None

    def validation_step(self, batch, batch_idx):
        input, input_weights, targets, targets_weights = batch
        pred = self(batch)
        result = {}
        loss = self.myloss(pred, targets[:, :, None].float(), targets_weights)
        result["val_loss"] = loss

        (
            valid_acc,
            valid_precision,
            valid_recall,
            valid_fscore,
            result_string,
        ) = calculate_metrics(pred, targets, targets_weights)
        result["precision"] = torch.tensor(valid_precision)
        result["recall"] = torch.tensor(valid_recall)
        return result

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([(x["val_loss"]) for x in outputs]).mean()
        avg_precision = torch.stack([(x["precision"]) for x in outputs]).mean()
        avg_recall = torch.stack([(x["recall"]) for x in outputs]).mean()

        # For tensorboard logging
        self.logger.experiment.add_scalar("Loss/Val", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar(
            "Precision/Val", avg_precision, self.current_epoch
        )
        self.logger.experiment.add_scalar("Recall/Val", avg_recall, self.current_epoch)

        # print(str(avg_precision) + " " + str(avg_recall))
        self.log("val_loss", avg_loss)
        return {"val_loss": avg_loss, "log": self.log}

    def test_step(self, batch, batch_idx):
        input, input_weights, targets, targets_weights = batch
        pred = self(batch)
        result = {}
        loss = self.myloss(pred, targets[:, :, None].float(), targets_weights)
        result["test_loss"] = loss

        (
            test_acc,
            test_precision,
            test_recall,
            test_fscore,
            result_string,
        ) = calculate_metrics(pred, targets, targets_weights)
        result["test_precision"] = torch.tensor(test_precision)
        result["test_recall"] = torch.tensor(test_recall)

        return result

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([(x["test_loss"]) for x in outputs]).mean()
        avg_precision = torch.stack([(x["test_precision"]) for x in outputs]).mean()
        avg_recall = torch.stack([(x["test_recall"]) for x in outputs]).mean()

        # For tensorboard logging
        self.logger.experiment.add_scalar("Loss/Test", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar(
            "Precision/Test", avg_precision, self.current_epoch
        )

        self.log("test_loss", avg_loss, "test_precision", avg_precision)
        return {"test_loss": avg_loss, "test_precision": avg_precision, "log": self.log}

    def myloss(self, pred, y, weight):
        mask = weight.view(-1, 1) * (
            (torch.reshape(torch.nn.BCELoss(reduction="none")(pred, y), [-1, 1]))
        )
        return torch.sum(mask) / torch.sum(weight).detach().item()

    def configure_optimizers(
        self,
    ):  # Pure Adam optimizer gives similar results as well.
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.001
        )
        return [optimizer]
