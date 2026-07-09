from modern_nlp.config import load_train_config
from modern_nlp.classification.dataset import load_dataset, prepare_dataset
from modern_nlp.classification.model import ClassificationModel
from modern_nlp.classification.trainer import ClassificationTrainer
from modern_nlp.classification.inference import ClassificationInference

train_config = load_train_config("modern_nlp/configs/train.yaml")
train_config.epochs = 1

raw_train, raw_val = load_dataset()
raw_train = raw_train.select(range(4))
raw_val = raw_val.select(range(4))

model = ClassificationModel(model_name="bert-base-uncased", num_labels=4)
train_dataset = prepare_dataset(raw_train, tokenizer=model.tokenizer, split_name="train_tmp")
val_dataset = prepare_dataset(raw_val, tokenizer=model.tokenizer, split_name="val_tmp")

trainer = ClassificationTrainer(model=model, train_dataset=train_dataset, eval_dataset=val_dataset, training_config=train_config)
trainer.train()
trainer.save_checkpoint(train_config.output_dir + "/final_tmp")

inf = ClassificationInference(train_config.output_dir + "/final_tmp")
print(inf.predict("Wall Street drops as tech stocks fall."))
