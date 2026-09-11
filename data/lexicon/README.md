# The lexicon goes here

Joi scores every emotional sentence with the **NRC Valence, Arousal, and Dominance Lexicon v2.1** by Saif M.
Mohammad (National Research Council Canada). It is free for research and non-commercial use under its own terms, and
it is not redistributed with this project.

1. Download it from https://saifmohammad.com/WebPages/nrc-vad.html (accept the terms of use on that page).
2. Unzip, and copy the file `NRC-VAD-Lexicon-v2.1.txt` (the one at the top level of the archive, 54 800 terms,
   four tab-separated columns: term, valence, arousal, dominance) into this folder:

```
data/lexicon/NRC-VAD-Lexicon-v2.1.txt
```

That is the path in `config.json` (`lexicon_file`). Nothing else is needed; the server refuses to start with a clear
message until the file is in place.

Please cite the lexicon if you publish anything built on this:

> Saif M. Mohammad. Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words.
> Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL), 2018.
