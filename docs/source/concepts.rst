.. _concepts:

Background and Concepts
========================================

The idea for GPTPipelines came from a research project where my team wanted to create tabular data about `pest outbreaks <https://en.wikipedia.org/wiki/Choristoneura_fumiferana>`__ to use in a model, but most all of the outbreak information that we could find was written as natural language in research papers. My team created some tabular data by hand, but the process of hand-labelling data takes a very long time, and therefore isn't scalable. There were *thousands* of papers about these outbreaks available online, and going through each one to pick out specific data is unreasonable.

So, how do we squeeze tabular data out of thousands of texts? Well, we can have ChatGPT do it. The idea is simple--gather all of your texts in one place, and have ChatGPT read each one, plus an instruction prompt, and record the output in a table. This works pretty well for simple questions, like "Is this text about Spruce Budworms?" or "Is this text in English?", but more complicated queries require more work. 