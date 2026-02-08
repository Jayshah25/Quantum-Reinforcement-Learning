.. qrl-qai documentation master file, created by
   sphinx-quickstart on Thu Aug 28 15:11:37 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to qrl-qai’s documentation!
===================================

**qrl-qai** is a python framework built on top of **Gymnasium**, **PennyLane**, and **PyTorch** to serve as a central platform for everything **quantum reinforcement learning**.

The current release (0.3.0) offers 5 native quantum RL style environments:

* **BlochSphereV0**
* **CompilerV0**
* **ErrorChannelV0**
* **ExpressibilityV0**
* **ProbabilityV0**

Check out the Installation and Quickstart guides to get started.

Additionally, each significant release has an asssociated Google Colab and Lightning AI Studio for a hassle free experience. These are especially useful for users who want to quickly test out the environments without going through the installation process.

Lightnining AI Studio contains a Streamlit playground for no-code experimentation with the environments!

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Version
     - Google Colab
     - Lightning AI Studio
   * - **0.1.0**
     - \-
     - \-
   * - **0.2.0**
     - |colab|
     - |lightning|
   * - **0.3.0**
     - \-
     - \-


.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1vtPB5_KRVkD3-4iiku4X8EmMpr_PoOY2?usp=sharing

.. |lightning| image:: https://img.shields.io/badge/_Open_in_Lightning_AI-792EE5?logo=lightning&logoColor=white
   :target: https://lightning.ai/jayshah25/studios/qrl-qai-0-2-0-playground


.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: BASICS

   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: TUTORIALS

   tutorials/bloch_sphere
   tutorials/probability
   tutorials/error_channel
   tutorials/expressibility
   tutorials/compiler

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: ENVIRONMENTS

   api/base_env
   api/blochsphere
   api/compiler
   api/errorchannel
   api/expressibility
   api/probability

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: AGENTS

   api/agents
   .. api/utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`