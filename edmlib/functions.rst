.. _functions:

****************
EDMLib Functions
****************

Grade Data
==========

.. autoclass:: edmlib.gradeData

.. _gradeDataInit:

Initialization
--------------

.. autofunction:: edmlib.gradeData.__init__

.. autofunction:: edmlib.gradeData.defineWorkingColumns

Functions
---------

Filtering / Getting Data
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: edmlib.gradeData.dropMissingValuesInColumn

.. autofunction:: edmlib.gradeData.filterByGpaDeviationMoreThan

.. autofunction:: edmlib.gradeData.filterColumnToValues

.. autofunction:: edmlib.gradeData.filterToMultipleMajorsOrClasses

.. autofunction:: edmlib.gradeData.filterStudentsByMajors

.. autofunction:: edmlib.gradeData.getCorrelationsWithMinNSharedStudents

.. autofunction:: edmlib.gradeData.getColumn

.. autofunction:: edmlib.gradeData.getDictOfStudentMajors

.. autofunction:: edmlib.gradeData.getListOfClassCodes

.. autofunction:: edmlib.gradeData.getUniqueIdentifiersForSectionsAcrossTerms

.. autofunction:: edmlib.gradeData.getNormalizationColumn

.. autofunction:: edmlib.gradeData.getGPADeviations

.. autofunction:: edmlib.gradeData.getGPAMeans

.. autofunction:: edmlib.gradeData.getPandasDataFrame

.. autofunction:: edmlib.gradeData.substituteSubStrInColumn

Export
^^^^^^

.. autofunction:: edmlib.gradeData.exportCSV

.. autofunction:: edmlib.gradeData.exportCorrelationsWithMinNSharedStudents

.. autofunction:: edmlib.gradeData.gradePredict

.. autofunction:: edmlib.gradeData.instructorRanks

.. autofunction:: edmlib.gradeData.instructorRanksAllClasses

.. autofunction:: edmlib.gradeData.outputGpaDistribution

.. autofunction:: edmlib.gradeData.sankeyGraphByCourseTracks

.. autofunction:: edmlib.gradeData.sankeyGraphByCourseTracksOneGroup


Logging / Troubleshooting
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: edmlib.gradeData.printColumn

.. autofunction:: edmlib.gradeData.printUniqueValuesInColumn

.. autofunction:: edmlib.gradeData.printEntryCount

.. autofunction:: edmlib.gradeData.printFirstXRows

.. _correlation:

Correlational Data
==================

.. autoclass:: edmlib.classCorrelationData

Initialization
--------------

.. autofunction:: edmlib.classCorrelationData.__init__

Functions
---------

Filtering / Getting Data
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: edmlib.classCorrelationData.dropMissingValuesInColumn

.. autofunction:: edmlib.classCorrelationData.filterColumnToValues

.. autofunction:: edmlib.classCorrelationData.filterToMultipleMajorsOrClasses

.. autofunction:: edmlib.classCorrelationData.getCliques

.. autofunction:: edmlib.classCorrelationData.getNxGraph

.. autofunction:: edmlib.classCorrelationData.substituteSubStrInColumn

.. _correlationExport:

Export / Graphs
^^^^^^^^^^^^^^^

.. autofunction:: edmlib.classCorrelationData.exportCSV

.. autofunction:: edmlib.classCorrelationData.chordGraphByMajor

.. autofunction:: edmlib.classCorrelationData.outputCliqueDistribution

Logging / Troubleshooting
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: edmlib.classCorrelationData.printClassesUsed

.. autofunction:: edmlib.classCorrelationData.printMajors

.. autofunction:: edmlib.classCorrelationData.printEntryCount

.. autofunction:: edmlib.classCorrelationData.printFirstXRows