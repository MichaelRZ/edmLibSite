.. _commonUses:

*********************
Common Usage Examples
*********************

Importing the Fordham Dataset
=============================

This requires approval by the department first. After that, write a python file in the same directory as the dataset, 
with the correct file name:

.. tabs::

    .. code-tab:: python Python

        from edmlib import gradeData, classCorrelationData

        df = gradeData("fordhams_dataset_fileName.csv")
        df.defineWorkingColumns('OTCM_FinalGradeN', 'SID', 'REG_term', 'REG_CourseCrn', 
				'REG_Programcode', 'REG_Numbercode', 'GRA_MajorAtGraduation')

Filtering Data to Certain Classes or Majors
===========================================

For majors, define a python list with major or department names, making sure they are spelled the same in 
the defined 'classDept' column (this is the 'REG_Programcode' column in the fordam dataset).

For classes, define a python list with class names that match the defined 'classCode' column. If 'classCode' was 
not defined but instead both 'classDept' and 'classNumber' were (like in the fordham dataset), these columns are added to define the 'classCode' column automatically, so use the concatenation of the two columns 
(e.g. 'Psych1000' from 'Psych' and '1000'). 

This example filters to Computer Science classes and "core" classes, as defined by Fordham.

.. tabs::

    .. code-tab:: python Python

        from edmlib import gradeData, classCorrelationData

        df = gradeData("fordhams_dataset_fileName.csv")
        df.defineWorkingColumns('OTCM_FinalGradeN', 'SID', 'REG_term', 'REG_CourseCrn', 
				'REG_Programcode', 'REG_Numbercode', 'GRA_MajorAtGraduation')

        majorsToFilterTo = ['Computer and Info Science', 
                            'Psychology']
        coreClasses = [ 'Philosophy1000',
                        'Theology1000',
                        'English1102',
                        'English1101',
                        'History1000',
                        'Theology3200',
                        'VisualArts1101',
                        'Physics1201',
                        'Chemistry1101']

        df.filterToMultipleMajorsOrClasses(majorsToFilterTo, coreClasses)

A similar filter is available for the correlational data class:

.. tabs::

    .. code-tab:: python Python

        data = classCorrelationData('outputCorrelation.csv')
        data.filterToMultipleMajorsOrClasses(majorsToFilterTo, coreClasses)

Filtering Data to Students of Specific Majors
=============================================

For filtering data to students who have declared certain majors, the 'studentMajor' column should have been 
defined with 'defineWorkingColumns'. It should be noted that if a student ever declared one of these majors, 
they will be included. The syntax is very similar:

.. tabs::

    .. code-tab:: python Python

        from edmlib import gradeData, classCorrelationData

        df = gradeData("fordhams_dataset_fileName.csv")
        df.defineWorkingColumns('OTCM_FinalGradeN', 'SID', 'REG_term', 'REG_CourseCrn', 
				'REG_Programcode', 'REG_Numbercode', 'GRA_MajorAtGraduation')

				# function takes a 'list'
				df.filterStudentsByMajors(['Psychology', 'Economics'])

Exporting Correlation Data
==========================

Correlations can be obtained with the following example, which filters by GPA deviation first:

.. tabs::

    .. code-tab:: python Python

        from edmlib import gradeData, classCorrelationData

        df = gradeData('yourDataSet.csv')
        df.defineWorkingColumns('OTCM_FinalGradeN', 'SID', 'REG_term', 'REG_CourseCrn', 
				'REG_Programcode', 'REG_Numbercode', 'GRA_MajorAtGraduation')
        df.filterByGpaDeviationMoreThan(0.2)


        # The first parameter is the output file name, the second is the minimum 
        # number of students classes must share to calculate a correlation

        df.exportCorrelationsWithMinNSharedStudents('outputCorrelation.csv', 20)

Furthermore, this correlational data can be made into a chord graph like on the front page (:ref:`front`) 
by using the :ref:`correlation` class, which outputs to HTML and PNG in the same directory:

.. tabs::

    .. code-tab:: python Python

        data = classCorrelationData('outputCorrelation.csv')
        data.chordGraphByMajor()

This is currently limited to averaging data by major. Options for this function can be found here: 
:ref:`correlationExport`.