Installation
============

Install the package directly from PyPI:

.. code-block:: bash

   pip install qrl-qai


FFmpeg Requirement
------------------

To save the episodes as **mp4** videos, it is essential to have **ffmpeg** installed on your system.
Without ffmpeg, environments can run but episodes can only be saved as **gif**, not **mp4**.

To check if ffmpeg is installed (Windows / Linux / macOS):

.. code-block:: bash

   ffmpeg -version

To install ffmpeg:

Using Conda
~~~~~~~~~~~

You can install FFmpeg inside a conda environment:

.. code-block:: bash

   conda install -c conda-forge ffmpeg


Windows
~~~~~~~

Option 1: Using Chocolatey (recommended)

.. code-block:: bash

   choco install ffmpeg


Option 2: Manual installation

1. Visit:
   https://ffmpeg.org/download.html

2. Click **Windows**, then choose **gyan.dev** or **BtbN** builds.

3. Download the latest *full* build ZIP.

4. Extract it (for example, to ``C:\ffmpeg``).

5. Add ``C:\ffmpeg\bin`` to your system ``PATH``.


Linux
~~~~~

Debian / Ubuntu:

.. code-block:: bash

   sudo apt update
   sudo apt install ffmpeg -y


Fedora:

.. code-block:: bash

   sudo dnf install ffmpeg -y


Arch Linux:

.. code-block:: bash

   sudo pacman -S ffmpeg


macOS
~~~~~

Using Homebrew:

.. code-block:: bash

   brew install ffmpeg
