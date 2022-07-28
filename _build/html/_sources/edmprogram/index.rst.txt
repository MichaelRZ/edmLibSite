#########################
EDM Program Documentation
#########################

.. note::
		The program is still being developed and this page will be expanded accordingly.

This page documents usage of the EDM Program, which takes in data formatted as rows of grades received by students in their respective classes. The program is capable of filtering, sorting, and processing the data according to various parameters, as well as creating graphs of certain aspects such as GPA distribution or class grade correlations.

.. contents:: Contents 
		:local:

************
Installation
************

.. note::
		Installation in the future will be just downloading and running an executable depending on operating system. This python file approach is temporary.

1. Download both the :download:`library </edmlib.py>` and :download:`program </edmProgram.py>` python files and place them in the same directory.

2. Run the program with the command :code:`python3 edmProgram.py` in your computer's terminal.

3. Install any libraries needed if python gives errors saying they are missing. Current dependencies include:

* `Scipy <https://www.scipy.org/>`_ (includes pandas, numpy)
* `NetworkX <https://networkx.github.io/>`_ (Used for some graph functions)
* `HoloViews <http://holoviews.org/index.html>`_ (Used for visualizations)
* `PyQt5 <https://pypi.org/project/PyQt5/>`_ (Used for GUI)

**************
Importing Data
**************

To use the program, data first needs to be imported. Data should be organized in a CSV file, with each row corresponding to a grade a student has recieved in a particular class, and certain columns designated for terms, student IDs, and more, as detailed below.

The program will show the following when run:

.. image:: /images/import.png
   :width: 600

To open a file, go to file -> open and give it a second to load your CSV file. It should appear similar to this:

.. image:: /images/fileOpened.png
   :width: 600

Set Columns
-----------

To designate to the program which columns correspond to which data, there is a 'Set Columns' button on the tool bar. Click this, and there should be a list of options where you can designate your columns. This program requires certain columns be defined:

* Class ID - Identifiers specific to each class (different for different sections of the same class).
* Student ID - Identifiers specific to each student.
* Student Grade - Numeric grade recieved by a student, on a 1.0 - 4.0+ scale. (This may be expanded to accept other scales in the future)
* Term - Term a class was taken in. Ideally, this column is sortable, however an option will be made available to define term order.
* Student Major - Major declared by a student. Needed for major related functions.
* Class Credits - Number of credits a class counts for. Needed for more accurate GPA calculations.
* Faculty ID - Identifiers specific to instructors of the class.

Additionally, one of the following options are required:

Both,

* Class Department - The department the class in the row falls under.
* Class Number - A number given to a class for identification in a specific department.

Or,

* Class Code - Some specific name given to a class.

If a certain function of the program sees that one of these columns is needed and missing, an error will be given. These columns can be set with the 'Set Columns' button on the tool bar:

.. image:: /images/setcolumns.png
   :width: 600

They can also be set by right clicking an appropriate column and using the menu:

.. image:: /images/setColMenu.png
   :width: 600

Reset Dataset
-------------

Lastly, if you have done something with the program and find that you want to go back to the original dataset, there is a reload option under 'File -> Reload Original File'. There is also a save option in this menu.

.. image:: /images/filemenu.png
   :width: 600

*************************
Filtering / Altering Data
*************************

This section covers manipulation of the dataset, ranging from seeing an overview of what the file is holding to filtering and sorting rows according to certain criteria.

Show Stats
----------

A breif overview of the data can be seen with the 'Show Stats' option given in the toolbar, such as the number of unique values in each column:

.. image:: /images/stats.png
   :width: 600

Click 'Show Values' next to any of these columns and a window listing the unique values and their frequencies will appear:

.. image:: /images/statsdetail.png
   :width: 600

This window allows sorting and filtering by clicking / rightclicking the column headers as well:

.. image:: /images/statsdetailrc.png
   :width: 600

Column Operations
-----------------

Certain column operations can be done by right clicking a column. The operations available now are as follows:

* Set Column Menu - Designate columns to certain roles in the program.
* Rename Column - Renames a column to a given string.
* Substitute Menu - Operations relating to substituting values in the column.
* Filter / Numeric Operations Menu - Operations relating to filtering data and applying numerical rules.
* Drop Undefined Values in Column - Removes rows with undefined values in the column - used automatically when performing numeric operations.
* Delete Column - Deletes the column. The column can only be brought back by reloading the original file.

.. image:: /images/rcmenu.png
   :width: 600

Substitution
^^^^^^^^^^^^

The substitution menu has a few options:

* Substitute in Column - Substitute a given substring with something else.
* Substitute Many Values - Substitute several values in the column at once, and optionally save these substitutions to a file.
* Use substitution file - Substitute values in the column by using a file made by the 'Substitute Many Values' option.

.. image:: /images/substituteMenu.png
   :width: 600

Filtering / Numeric Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Filter / Numeric operations menu allows these operations:

* Filter Column to Values - Reduce the dataset to rows that contain one of the given values in the column.
* Filter Column Numerically - Filter the column with a given minimum and / or maximum value.
* Make Absolute Values - Convert numbers in the column to absolute values.
* Get Mean / Median / Mode - Displays mean, median, and mode of the column in the information bar. Only gets the mode if the column is non-numeric.
* Round Column - Round the underlying data in the column to a given number of decimal places. (Note: values are rounded to the third decimal place when viewing by default, but the data is left unrounded).

.. image:: /images/filterNumMenu.png
   :width: 600

Sorting
^^^^^^^

Columns can also be sorted, ascending or descending, by left-clicking the column header. Values can only be sorted numerically if the undefined values are removed first, which is available from the right-click menu.

.. image:: /images/colsort.png
   :width: 600

Filters
-------

Several filters are available, besides the column operations:

.. image:: /images/filtermenu.png
   :width: 600

Class Filter
^^^^^^^^^^^^

The classes / class department filter filters to rows that either are part of one of the given departments or are one of the given classes.

.. image:: /images/classdeptfilter.png
   :width: 600

Student Major Filter
^^^^^^^^^^^^^^^^^^^^

The student major filter filters to students who have ever declared one of the given majors.

.. image:: /images/majorfilter.png
   :width: 600

GPA Deviation Filter
^^^^^^^^^^^^^^^^^^^^

The class GPA deviation filter filters to classes that have above a given minimum standard deviation of GPA (Classes that have very similar grades between all students are dropped).

.. image:: /images/gpadevfilter.png
   :width: 600

Calculations
------------

Certain calculations that may be useful can be accessed with the calculations menu on the menu bar.

.. image:: /images/calculateMenu.png
   :width: 600

The first three options calculate the mean grade in the class, the standard deviation of grades in the class, and a normalized grade (number of standard deviations away from the mean the student is). These may be generated automatically when performing certain operations with the program.

Instructor ranking operations are also available in this menu.

******************
Ranking Professors
******************

Professors can be ranked by looking at the students who took their class, looking at those students' performance in a later class, and seeing how well they do compared to students who had different professors.

Across Pair of Classes
----------------------

This can be calculated for a certain pair of classes by selecting Calculations -> Calculate Instructor Effectiveness:

.. image:: /images/calculateMenu.png
   :width: 600

.. image:: /images/instructorRank.png
   :width: 600

This will generate a file and show the corresponding data (classes, grade benefits, and number of students used to calculate), which can be filtered and sorted:

.. image:: /images/rankPreview.png
   :width: 600

Across All Class Pairs
----------------------

Instructor effectiveness data can be calculated for all class pairs by selecting Calculations -> Calculate Instructor Effectiveness (All):

.. image:: /images/calculateMenu.png
   :width: 600

.. image:: /images/instructorRankAll.png
   :width: 600

This will generate the same data by instructor, but will also include columns for what the class used is and what later class was used.

*****************
Predicting Grades
*****************

The program can attempt to predict grades by looking at all past student performance, given a student's past grades and classes they want to predict performance for.

To try this, select the Grade Predict button on the menu bar, and the following window should appear:

.. image:: /images/gradePredict.png
   :width: 600

Here, past grades can be inputted, and classes to predict grades for can be chosen as well. Currently, there are two modes of prediction:

* Nearest Neighbor - Gives the grade in the future course of the most similar student on record according to the given past grades.
* Mean of Three Nearest - Gives the closest grade to the mean grade of the three most similar students on record according to the given past grades.

More modes will be added in the future, including machine learning methods. Once all the options are selected, hitting OK will display the predicted grades in the window's information bar:

.. image:: /images/predictResult.png
   :width: 600

*************
Making Graphs
*************

Options for some graphs are available in the "correlation" sub-menu. Directly from the original dataset, GPA Distribution graphs and course track distribution graphs can be made; these are the second and third options. The first option generates a class correlation file, used in the next section, and the fourth and fifth options are variations of the third option.

.. image:: /images/correlationmenu.png
   :width: 600

GPA Distribution Graph
----------------------

A GPA Distribution graph can be made by selecting Correlations -> Export GPA Distribution, then filling in the following options:

.. image:: /images/gpahist.png
   :width: 600

An HTML file with an interactive graph will be saved in the same directory, and will automatically open in a browser similar to this graph:

.. raw:: html
		:file: ../gpaHistogram.html

Course Track Distribution Graph
-------------------------------

Course track distribution graphs can also be made by selecting Correlations -> Export Course Track Graph, then filling in the options. The distribution of orders that students take the different classes will be made.

.. image:: /images/track.png
   :width: 600

Doing so will generate and open an HTML file again, in the form of an interactive Sankey graph similar to the following:

.. raw:: html
		:file: ../sankey.html

Selecting Correlations -> Export Course Track Graph (alternate) will open the following window, which makes the same graph but with groups of classes students can take in order:

.. image:: /images/track2.png
   :width: 600

Selecting Correlations -> Export Course Track Graph (Experimental) will open the following window, where terms can be numerically designated to take into account things like summer terms, and the previous window will prompt again:

.. image:: /images/track3.png
   :width: 600

******************
Correlational Data
******************

Correlational data, or the normalized correlation between grades of students between one class and another, can be generated by selecting Correlations -> Export Class Correlations.

.. image:: /images/correlationmenu.png
   :width: 600

.. image:: /images/classcorr.png
   :width: 600

This will take some time and generate a CSV file with correlations between pairs of classes. Open the resulting file with the program, and some options will change accordingly, with columns for two classes, the correlation, the p-value, and number of students shared available.

.. image:: /images/corrfile.png
   :width: 600

The file, stats, and column operation menus will stay the same, however the program no longer needs columns designated and has different filters and correlation operations for different graphs.


Filters
-------

Filters are available for correlational data to narrow the data down to certain classes or departments. The filters previously available by column are still present as well, such as constraints to numerical bounds or a set of values.

Column Operations
^^^^^^^^^^^^^^^^^

The same filters can be applied to columns:

.. image:: /images/corrcolumn.png
   :width: 600

Course Filter
^^^^^^^^^^^^^

A filter for classes and departments respective of both columns or not is available. Note, using departments will only work if class department and class number columns were defined when generating the data; it works by ignoring the numbers.

.. image:: /images/corrfilter.png
   :width: 600

.. image:: /images/corrdeptfilter.png
   :width: 600

Graphs
------

Different graphs are available for correlational data, for now clique distribution and correlation graphs by major (really just departments). These are available from the correlation menu again:

.. image:: /images/corrcorr.png
   :width: 600

Major Correlation Graph
^^^^^^^^^^^^^^^^^^^^^^^

For a correlation graph between majors (or departments), select Correlations -> Export Chord Graph by Major, which will show some options:

.. image:: /images/corrchord.png
   :width: 600

A chord graph will be saved and shown similar to the following depending on the parameters:

.. raw:: html
		:file: ../gradeGraph50.html

Clique Distribution Graph
^^^^^^^^^^^^^^^^^^^^^^^^^

For a histogram of cliques made by the correlations, or the sizes of connected graphs in the data given a correlation threshold, select Correlations -> Export Clique Histogram, which will give you some options:

.. image:: /images/corrclique.png
   :width: 600

This will generate, save, and load on a browser a graph similar to the following:

.. raw:: html
		:file: ../cliqueHistogram.html