.. _gettingStarted:

***************
Getting Started
***************

Installation
============

1. Download the library :download:`here </edmlib.py>` .
2. Place in the same directory as where you are writing your python file.
3. In your python file, write the following line at the top:

.. tabs::

   .. code-tab:: python Python

        from edmlib import gradeData, classCorrelationData

This gives us the classes with all the relevant methods. If python gives errors when compiling, make sure to install 
the libraries it mentions such as :obj:`numpy` or :obj:`pandas`.

Importing Data
==============

Grade data
----------

Your data needs to be in the form of a :code:`.csv` file in the same directory or a "pandas" dataframe. This 
library expects data in the form of a list of grades recieved by students in certain classes, with more 
information possibly usable in the future. The columns required for core functionality right now include 
columns for final grades, student ID's, class number or name (e.g. '1000' in "Psych 1000"), class major or 
department, and the term the class was held.

First, the data needs to be instantiated with the :code:`gradeData` class:

.. tabs::

   .. code-tab:: python Python

        data = gradeData('fileName.csv')
        
        # or, for a pandas dataframe,
        pandasData = gradeData(pandasDataFrame)

Then, either standard columns can be used for determining which column is which (where the data set has columns
:code:`finalGrade`, :code:`studentID`, :code:`term`, :code:`classID`, :code:`classDept`, and :code:`classNumber` all defined), 
or you define your own columns with the method :code:`defineWorkingColumns`. Here is an example with Fordham's 
dataset:

.. tabs::

   .. code-tab:: python Python

        df.defineWorkingColumns('OTCM_FinalGradeN', 'SID', 'REG_term', 'REG_CourseCrn', 
				'REG_Programcode', 'REG_Numbercode', 'GRA_MajorAtGraduation')

The order of the parameters is important here. For more details, such as how class names can be given without 
a department or number, see the class's :ref:`gradeDataInit` section of the :ref:`functions` page.

After that, all the functions on the page :ref:`functions` under :code:`gradeData` are ready for use.

Correlational data
------------------

If you have already used the library and exported correlational data with it, this data can also be imported in a 
similar way:

.. tabs::

   .. code-tab:: python Python

        data = classCorrelationData('fileName.csv')

Column names are standard within the program for this data and don't need to be changed.