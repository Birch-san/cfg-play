# CFG for LLMs

References:

- [paper: Stay on topic with Classifier-Free Guidance](https://arxiv.org/abs/2306.17806)
- [Transformers issue: Add Classifier-Free Guidance sampling](https://github.com/huggingface/transformers/issues/24536)
- [Transformers PR: add CFG for .generate()](https://github.com/huggingface/transformers/pull/24654)

## License

[`cfg_chat_play`](scripts/cfg_chat_play.py), [`callback_text_iterator_streamer`](src/callback_text_iterator_streamer.py) and [`mps_type_monkeypatch`](src/mps_type_monkeypatch.py) are released under BSD 3-clause license, copyright 2023 Alex Birch.

[`cfg_chat_play`](scripts/cfg_chat_play.py) includes MIT-licensed code copied from Artidoro Pagnoni's [qlora](https://github.com/artidoro/qlora) and [Apache-licensed](licenses/MosaicML-mpt-7b-chat-hf-space.Apache.LICENSE.txt) code copied from MosaicML's [mpt-7b-chat](https://huggingface.co/spaces/mosaicml/mpt-7b-chat/blob/main/app.py) Huggingface Space.