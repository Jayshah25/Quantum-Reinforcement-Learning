.. qrl-qai documentation master file, created by
   sphinx-quickstart on Thu Aug 28 15:11:37 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to qrl-qai’s documentation!
===================================

**qrl-qai** is a python framework built on top of **Gymnasium**, **PennyLane**, and **PyTorch** to serve as a central platform for everything **quantum reinforcement learning**.

The current release (1.0.0) offers 2 RL algorithms (ValueIteration and QValueIteration) and 6 native quantum RL style environments (BlochSphereV0, BlochSphereV1, CompilerV0, ErrorChannelV0, ExpressibilityV0, ProbabilityV0).

Check out the Installation and Quickstart guides to get started.

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
   * - **1.0.0**
     - |colab1|
     - |lightning1|


.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1vtPB5_KRVkD3-4iiku4X8EmMpr_PoOY2?usp=sharing

.. |colab1| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1wsThxxrvHh0Vboftay-eFbCrdVzKO-I3?usp=sharing


.. |lightning| image:: https://img.shields.io/badge/_Open_in_Lightning_AI-792EE5?logo=lightning&logoColor=white
   :target: https://lightning.ai/jayshah25/studios/qrl-qai-0-2-0-playground

.. |lightning1| image:: https://img.shields.io/badge/_Open_in_Lightning_AI-792EE5?logo=lightning&logoColor=white
   :target: https://lightning.ai/jayshah25/templates/qrl-qai-playground-1-0-0



Additionally, each significant release has an asssociated Google Colab and Lightning AI Studio for a hassle free experience. These are especially useful for users who want to quickly test out the environments without going through the installation process.

Lightnining AI Studio contains a Streamlit playground for no-code experimentation with the environments!



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

   tutorials/value_iteration
   tutorials/qvalue_iteration
   tutorials/bloch_sphere_v0
   tutorials/bloch_sphere_v1
   tutorials/compiler
   tutorials/error_channel
   tutorials/expressibility
   tutorials/probability


.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: ALGORITHMS

   api/base_iteration
   api/value_iteration
   api/qvalue_iteration

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: ENVIRONMENTS

   api/base_env
   api/blochsphereV0
   api/blochsphereV1
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