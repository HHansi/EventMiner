# Created by Hansi at 5/2/2021
from sentence_transformers import SentenceTransformer, InputExample, losses, models, util
from sentence_transformers import evaluation
from sklearn.metrics.pairwise import paired_cosine_distances
from torch.utils.data import DataLoader

from algo.config.coref_args import SPArgs


class STRegressionModel():  # Sentence-Transformer Regression Model (Semantic Textual Similarity Task)
    def __init__(
            self,
            model_name,
            args=None,
            embedding_learning=None
    ):
        """
        Initializes a STClassificationModel
        :param model_name:
        :param args:
        """
        self.args = SPArgs()

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, SPArgs):
            self.args = args

        if embedding_learning is not None and embedding_learning == 'from-scratch':
            word_embedding_model = models.Transformer(model_name, max_seq_length=self.args.max_seq_length)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        else:
            self.model = SentenceTransformer(model_name)

    def train(self, train_df, eval_df):
        """

        :param train_df: dataframe with columns 'text_a', 'text_b', 'labels'
        :param eval_df: dataframe with columns 'text_a', 'text_b', 'labels'
        :return:
        """

        # format training data
        if "text_a" in train_df.columns and "text_b" in train_df.columns and "labels" in train_df.columns:
            if self.args.do_lower_case:
                train_df.loc[:, 'text_a'] = train_df['text_a'].str.lower()
                train_df.loc[:, 'text_b'] = train_df['text_b'].str.lower()

            train_examples = [
                InputExample(str(i), [text_a, text_b], label)
                for i, (text_a, text_b, label) in enumerate(
                    zip(
                        train_df["text_a"].astype(str),
                        train_df["text_b"].astype(str),
                        train_df["labels"].astype(float),
                    )
                )
            ]
        else:
            raise KeyError('Training data processing - Required columns not found!')

        # format evaluation data
        if "text_a" in train_df.columns and "text_b" in train_df.columns and "labels" in eval_df.columns:
            if self.args.do_lower_case:
                eval_df.loc[:, 'text_a'] = eval_df['text_a'].str.lower()
                eval_df.loc[:, 'text_b'] = eval_df['text_b'].str.lower()

            evaluator = evaluation.EmbeddingSimilarityEvaluator(list(eval_df["text_a"]), list(eval_df["text_b"]),
                                                                list(eval_df["labels"]), batch_size=self.args.eval_batch_size)
        else:
            raise KeyError('Evaluation data processing - Required columns not found!')

        # Define train dataset, the dataloader and the train loss
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.args.train_batch_size)
        train_loss = losses.CosineSimilarityLoss(self.model)

        # Tune the model
        self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=self.args.num_train_epochs,
                       warmup_steps=self.args.warmup_steps, optimizer_params={'lr': self.args.learning_rate},
                       weight_decay=self.args.weight_decay, evaluator=evaluator,
                       evaluation_steps=self.args.evaluate_during_training_steps,
                       max_grad_norm=self.args.max_grad_norm, output_path=self.args.best_model_dir,
                       show_progress_bar=self.args.show_progress_bar)

    def predict(self, data_df):
        if "text_a" in data_df.columns and "text_b" in data_df.columns:
            if self.args.do_lower_case:
                data_df.loc[:, 'text_a'] = data_df['text_a'].str.lower()
                data_df.loc[:, 'text_b'] = data_df['text_b'].str.lower()

            # Compute embedding for text pairs
            embeddings1 = self.model.encode(list(data_df["text_a"]), convert_to_numpy=True)
            embeddings2 = self.model.encode(list(data_df["text_b"]), convert_to_numpy=True)

            cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
            return cosine_scores
        else:
            raise KeyError('Prediction data processing - Required columns not found!')