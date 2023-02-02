# Attributed Question Answering

Attributed Question Answering (QA) as a key first step in the development of
attributed LLMs. There are multiple possible motivations for studying Attributed
QA:

1. It is perhaps the simplest possible information-seeking application, and as
such it is relatively straightforward to evaluate Attributed QA systems.
2. In spite of its simplicity, models and experiments for Attributed QA are
likely to be highly informative to the general goal of building attributed LLMs.
3. Attributed QA is an interesting task that has advantages over existing
approaches to the evaluation of QA systems.


In Attributed QA, the input is a question, and the output is an `(answer,
attribution)`  pair where `answer` is an answer string, and `attribution` is a
pointer into a fixed underlying corpus. The attribution should give supporting
evidence for the answer; for example, it should satisfy the conditions of
[AIS](https://arxiv.org/abs/2112.12870).


***what is the population of st petersburg fl***

| Answer  | Attribution |
| ------- | ----------- |
| 244,769 | According to the 2010 census, the city contained **244,769** people, making St. Petersburg the largest city in Pinellas County, and 129,401 households. The population density was 3,964.4 per square mile (1530.7/km2). [Title: [St. Petersburg, Florida](https://en.wikipedia.org/wiki/St._Petersburg,_Florida) Section: Demographics, 2010 Census] |
| 263,768 | St. Petersburg, Florida is the fifth largest city in Florida with a population of **263,768** as of 2017. The city is home to 74 completed high rises (as of 2018), and the most notable are the One St. Petersburg, Priatek Plaza and Signature Place skyscrapers. [Title: [List of tallest buildings in St. Petersburg, Florida](https://en.wikipedia.org/wiki/List_of_tallest_buildings_in_St._Petersburg,_Florida)] |

We include both data and code to support research into Attributed QA in this
release. If you use this in your work, please cite our paper.

```
@misc{https://doi.org/10.48550/arxiv.2212.08037,
  doi = {10.48550/ARXIV.2212.08037},
  url = {https://arxiv.org/abs/2212.08037},
  author = {Bohnet, Bernd and Tran, Vinh Q. and Verga, Pat and Aharoni, Roee and Andor, Daniel and Soares, Livio Baldini and Ciaramita, Massimiliano and Eisenstein, Jacob and Ganchev, Kuzman and Herzig, Jonathan and Hui, Kai and Kwiatkowski, Tom and Ma, Ji and Ni, Jianmo and Saralegui, Lierni Sestorain and Schuster, Tal and Cohen, William W. and Collins, Michael and Das, Dipanjan and Metzler, Donald and Petrov, Slav and Webster, Kellie},
  title = {Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}

```

## Data

### Attribution Corpus

For comparability between systems, our paper standardizes the collection of
allowable attributions to the provided scrape of Wikipedia release, taken on
2021-10-13 and processed with [Pyserini](https://github.com/castorini/pyserini).
Access to this data is via [Google Cloud](https://storage.googleapis.com/gresearch/attributed_language_models/wikipedia.zip).
You will need to unzip the downloaded file and note the location of the unzipped
files, for use as the `--wikipedia_glob` argument in the evaluation script.

```
« St. Elsewhere » « St. Elsewhere, Episodes, "Their Town" » In a somewhat change-of-pace episode, Drs. Craig and Novino, Ellen Craig, and Lizzie Westphall visit Donald and Tommy Westphall (Lizzie\'s father and brother, respectively), who appear to be enjoying the quiet life in small town New Hampshire. The episode features Dr. Westphall occasionally breaking the fourth wall and speaking directly to the viewer, a la the "Stage Manager" character in Our Town (the episode title and its location are nods to the Thornton Wilder play).
```

### System Evaluation

Human and automatic evaluation of system output is provided in `ratings.csv`:

### Dataset Statistics

| Statistic | |
|--------------|--------|
| Dataset size | 67.7MB |
| Number of instances | 83,030 (3610 examples x 23 systems) |
| Number of fields | 8, described below |
| Human labels | 23,000 (1000 examples x 23 systems) |
| Automatic labels | 83,030 (3610 examples x 23 systems) |

### Dataset Structure

The format is columns headed and containing:

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `system_name` | `string` | System identifier from Tables 1, 2, and 3 of the paper. | Post-4 |
| `question` | `string` | A question from the development set of OpenNQ | who played hyde in league of extraordinary gentlemen |
| `answer` | `string` | The system-generated answer span | Jason Flemyng |
| `attribution` | `string` | An identifier from the Attribution Corpus | `http://en.wikipedia.org/wiki/Jason_Flemyng#Jason_Flemyng#Television_and_film_work#2` |
| `passage` | `string` | The corresponding passage from the Attribution Corpus formatted by `evaluation.py` | Title: Jason Flemyng Section: Television and film work In the early 2000s he featured in two big-budget Hollywood films which were adaptations of Alan Moore comic books; as John Netley in 2001's From Hell, with Johnny Depp, and 2003's The League of Extraordinary Gentlemen, with Sean Connery, in which Flemyng played Dr. Henry Jekyll and Edward Hyde. The latter film was a disappointment, but Flemyng commented that: ""It was a bit of a nightmare... the film cost a fortune and didn't make back the money it was meant to... But I still get a huge kick out of doing films like that and From Hell. Any day you walk onto a set and Sean Connery or Johnny Depp or Brad Pitt is there has to be a good day. |
| `human_rating` | `Y / N` | The attribution decision of human rating according to 5-way annotation | Y |
| `auto_ais` | `Y / N` | The attribution decision of AutoAIS, Y for attributable (`nli_score` score greater than 0.5), or N. | Y |
| `nli_score` | `float` | The entailment score of the AutoAIS model | 0.9814687 |


### Languages

This release is in English.


## Evaluation Script

Automatic evaluation of Attributed QA performs AutoAIS and SQuAD EM scoring
over an input predictions .csv (provided via the `--predictions_file` argument),
where the columns are headed and contain:

* `question` a question from NQ Open (questions which do not match will be discarded from evaluation)
* `answer` an output string
* `attribution` an index from the Attribution Corpus (indexes which do not match will be discarded from evaluation)

```
who played hyde in league of extraordinary gentlemen,Jason Flemyng,http://en.wikipedia.org/wiki/Jason_Flemyng#Jason_Flemyng#Television_and_film_work#2
```

The results in the paper are for short-answer seeking queries in the [Natural
Questions](https://research.google/pubs/pub47761/). `evaluation.py` supports
analysis of the **Validation Set** for which we use the version [`natural_questions_open:1.0.0`](https://www.tensorflow.org/datasets/catalog/natural_questions_open) from `tensorflow_datasets` of [Open-NQ](https://aclanthology.org/P19-1612/).

The output of `evaluation.py` is two files, provided by the arguments:

* `--scores_file` A short file with evaluation scores to report. To compare to
  our paper, see especially **AutoAIS** and **SQuAD (em)**.
* `--ais_output_file` A table with rows of **question**, **answer** (strings
from `--predictions_file`), and **passage** (the retrieved passage string
 from the Attribution Corpus, formatted for human assessment of AIS. **autoais**
 gives the automatic judgment of attribution, Y or N.
