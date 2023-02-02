# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
"""Evaluation of Attributed QA systems.

The input file format is a .csv of three columns with headers:
  question,answer,attribution

where the attribution is an identifier in the specified Wikipedia corpus.
Evaluation proceeds using NLI classification, per:
  https://arxiv.org/pdf/2204.04991.pdf

The model to use is: google/t5_xxl_true_nli_mixture
"""
from collections.abc import Sequence
from collections import defaultdict
import csv
import functools
import glob
import json
import multiprocessing
import re

from absl import app
from absl import flags
from absl import logging
from t5.evaluation.metrics import squad
import tensorflow_datasets as tfds
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer

# Constants that are needed to ensure consistency with Attributed QA paper.
NQ_OPEN = "natural_questions_open:1.0.0"
NQ_SPLIT = "validation"
AUTOAIS = "google/t5_xxl_true_nli_mixture"

PASSAGE_FORMAT = re.compile("« ([^»]*) » « ([^»]*) » (.*)")

# File locations.
flags.DEFINE_string("predictions_file", "", "Path to system predictions.")
flags.DEFINE_string("scores_file", "",
                    "Path for file to write evaluation scores.")
flags.DEFINE_string("ais_output_file", "",
                    "Path for file to write output for later AIS analysis.")
flags.DEFINE_string("wikipedia_glob", "", "Glob for Wikipedia corpus.")
flags.DEFINE_integer("processes", 16,
                     "Number of processes to use in reading Wikipedia.")
FLAGS = flags.FLAGS


def read_nq_answers():
  """Reads NQ Open from tensorflow_datasets."""
  nq_answers = {}
  for example in tfds.load(NQ_OPEN)[NQ_SPLIT]:
    question = example["question"].numpy().decode()
    answers = [a.decode() for a in example["answer"].numpy()]
    nq_answers[question] = answers
  return nq_answers


def read_wikipedia(shard, predicted_ids):
  """Reads one shard of Wikipedia dump and returns dictionary of passages."""
  passages = {}
  with open(shard, mode="r") as f:
    for line in f:
      try:
        d = json.loads(line)
      except ValueError:
        logging.error("Failed parsing JSON line.")
        continue
      if d["id"] in predicted_ids:
        passages[d["id"]] = d["contents"]
  return passages


def format_passage_for_autoais(passage):
  """Produce the NLI format for a passage.

  Args:
    passage: A passage from the Wikipedia scrape.

  Returns:
    a formatted string, e.g.

      Luke Cage (season 2), Release. The second season of Luke Cage was released
      on June 22, 2018, on the streaming service Netflix worldwide, in Ultra HD
      4K and high dynamic range.
  """
  m = PASSAGE_FORMAT.match(passage)
  if not m:
    return passage

  headings = m.group(2)
  passage = m.group(3)
  return f"{headings}. {passage}"


def format_passage_for_ais(passage):
  """Produce the AIS template format of a passage.

  Args:
    passage: A passage from the Wikipedia scrape.

  Returns:
    a formatted string, e.g.

      Title: Luke Cage (season 2)
      Section: Release

      The second season of Luke Cage was released on June 22, 2018, on the
      streaming service Netflix worldwide, in Ultra HD 4K and high dynamic
      range.
  """
  ret = []

  m = PASSAGE_FORMAT.match(passage)
  if not m:
    return passage

  title = m.group(1)
  ret.append(f"Title: {title}")

  headings = m.group(2)
  if not headings.startswith(title):
    # In rare cases, the section does not start with the title, e.g.
    # « Chrysler 300 letter series » « First Generation, 1955 C-300 » ...
    sections = m.group(2)
    ret.append(f"Section: {sections}")
  elif len(headings) > len(title):
    sections = headings[len(title) +
                        2:]  # 2 characters follow the heading: ", "
    ret.append(f"Section: {sections}")
  ret.append("")

  passage = m.group(3)
  ret.append(passage)
  return "\n".join(ret)


def format_example_for_autoais(example):
  return "premise: {} hypothesis: The answer to the question '{}' is '{}'".format(
      format_passage_for_autoais(example["passage"]), example["question"],
      example["answer"])


def infer_autoais(example, tokenizer, model):
  """Runs inference for assessing AIS between a premise and hypothesis.

  Args:
    example: Dict with the example data.
    tokenizer: A huggingface tokenizer object.
    model: A huggingface model object.

  Returns:
    A string representing the model prediction.
  """
  input_text = format_example_for_autoais(example)
  input_ids = tokenizer(input_text, return_tensors="pt").input_ids
  outputs = model.generate(input_ids)
  result = tokenizer.decode(outputs[0], skip_special_tokens=True)
  inference = "Y" if result == "1" else "N"
  example["autoais"] = inference
  return inference


def score_predictions(predictions, nq_answers):
  """Scores model predictions against AutoAIS and NQ answers.

  Args:
    predictions: A dict from questions to prediction rows.
    nq_answers: A dict from questions to lists of NQ reference answers.
    passages: A dict from identifiers from the attribution corpus to the
      corresponding paragraphs.

  Returns:
    a dict of metric values, keyed by metric names
  """
  hf_tokenizer = T5Tokenizer.from_pretrained(AUTOAIS)
  hf_model = T5ForConditionalGeneration.from_pretrained(AUTOAIS)

  autoais = 0
  target_answers = []
  predicted_answers = []
  for question, answers in nq_answers.items():
    target_answers.append(answers)
    example = predictions.get(question, None)
    if example is None:
      logging.error("Did not find prediction for '%s'", question)
      predicted_answers.append("")
      continue
    predicted_answers.append(example["answer"])
    if not example["passage"]:
      continue
    inference = infer_autoais(example, hf_tokenizer, hf_model)
    autoais += inference == "Y"

  scores = {}
  scores["AutoAIS"] = autoais / len(target_answers)
  for metric, score in squad(target_answers, predicted_answers).items():
    scores[f"SQuAD ({metric})"] = score
  return scores


def main(unused_argv: Sequence[str]) -> None:
  del unused_argv

  logging.info("Loading NQ...")
  nq_answers = read_nq_answers()
  logging.info("Loaded %d answers from NQ." % len(nq_answers))

  logging.info("Reading predictions...")
  predictions = {}
  predicted_ids = set()
  with open(FLAGS.predictions_file, mode="r") as f:
    reader = csv.DictReader(f)
    for row in reader:
      question = row["question"]
      if question not in nq_answers:
        logging.warning("Skipping '%s' - not found in NQ!", question)
      else:
        predictions[question] = row
        predicted_ids.add(row["attribution"])
  read_wikipedia_fn = functools.partial(
      read_wikipedia, predicted_ids=predicted_ids)

  logging.info("Retrieving passages...")
  passages = {}
  pool = multiprocessing.Pool(FLAGS.processes)
  for p in pool.imap_unordered(read_wikipedia_fn,
                               glob.glob(FLAGS.wikipedia_glob)):
    passages.update(p)
  pool.close()
  pool.join()
  for example in predictions.values():
    example["passage"] = passages.get(example["attribution"], "")

  logging.info("Scoring predictions...")
  scores = score_predictions(predictions, nq_answers)

  logging.info("Writing outputs...")
  with open(FLAGS.scores_file, mode="w") as f:
    for metric, score in scores.items():
      logging.info("%s: %f", metric, score)
      f.write(f"{metric}: {score}\n")
  with open(FLAGS.ais_output_file, mode="w") as f:
    writer = csv.DictWriter(f, fieldnames=["question", "answer", "passage", "autoais"])
    writer.writeheader()
    for example in predictions.values():
      row = {
          "question": question,
          "answer": example["answer"],
          "passage": format_passage_for_ais(example["passage"]),
          "autoais": example.get("autoais", "N")
      }
      writer.writerow(row)


if __name__ == "__main__":
  app.run(main)
