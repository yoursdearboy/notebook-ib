# Research

I see three main approaches, though there may be more.
The solution doesn't have to be exactly of one type.

- **Notebook (ref. NotebookLM)** – information retrieval and tools around this info;
- **AI tutor** – maintains user's knowledge model;
- **(Personal) Knowledge Base (wiki)** – another way to organize and retrieve information.

Below mostly links and references:
⭐️ interesting
🤔 assess

## Notebook

Notebook built on RAG, with optional web search and other tools (see below).

- Google NotebookLM ⭐️

   [Website](https://notebooklm.google.com)

   Available tools:

   <img width="542" height="185" alt="image" src="https://github.com/user-attachments/assets/be426254-04f1-4103-aa02-a371dacf4ba9" />

- OpenNotebook ⭐️

   [Website](https://www.open-notebook.ai)
   [lfnovo/open-notebook](https://github.com/lfnovo/open-notebook)

   Open source clone of NotebookLM with more features.

- SurfSense

  [Website](https://www.surfsense.com)
  [MODSetter/SurfSense](https://github.com/MODSetter/SurfSense)

## AI tutor

It has memory of a student's knowledge, gaps, interests, grades.

- Khanmigo ⭐️

  [Website](https://www.khanmigo.ai)

  By Khan Academy (and Microsoft?[^2]).
  Free for teachers worldwide, paid for students or through Khan Academy.
  Socratic learning approach (?)
  See also [^1]

- Mr. Ranedeer: Your personalized AI Tutor

  [JushBJJ/Mr.-Ranedeer-AI-Tutor](https://github.com/JushBJJ/Mr.-Ranedeer-AI-Tutor)
  Discontinued

  A popular prompt for better personalized learning experience.

- DeepTutor 🤔

  [Website](https://deeptutor.info)
  [HKUDS/DeepTutor](https://github.com/HKUDS/DeepTutor)

  Full featured and complex agent with UI. Also see the paper [^3]

## (Personal) Knowledge Base

Personal wiki with linked materials, with or without RAG.

- LLM Wiki (Andrej Karpathy) 🤔

  [karpathy/llm-wiki.md](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)

  Very simple approach to AI driven personal knowledge base using only local files and any agent the user already has.

- Openwiki

  [langchain-ai/openwiki](https://github.com/langchain-ai/openwiki)

  For code, but still interesting.

- WeKnora

  [Tencent/WeKnora](https://github.com/Tencent/WeKnora)

  RAG-based wiki, but designed for enterprise, not personal use.



[^1]: [How AI Could Save (Not Destroy) Education | Sal Khan | TED](https://www.youtube.com/watch?v=hJP5GqnTrNo)
[^2]: [Enhancing the future of education with Khan Academy](https://www.microsoft.com/en-us/education/blog/2024/05/enhancing-the-future-of-education-with-khan-academy/)
[^3]: [DeepTutor: Towards Agentic Personalized Tutoring, Bingxi Zhao et al](https://arxiv.org/abs/2604.26962)
