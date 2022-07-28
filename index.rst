
.. edmLibSite documentation master file, created by
   sphinx-quickstart on Thu May 28 10:53:43 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _front:

======
EDMLib
======

This site was made for Fordham University's "Educational Data Mining" research. There are links to the associated 
library and program, as well as documentation, below.

Downloads
=========

.. include:: edmLibDownload.inc

.. include:: edmProgramDownload.inc

.. toctree::
   :maxdepth: 1

   edmlib/index
   edmprogram/index

.. raw:: html

		<br>

Visualizations
==============

.. raw:: html

		<!-- Tab links -->
		<div class="tab">
			<button class="tablinks" onclick="openCity(event, 'GPAs')">GPAs</button>
			<button class="tablinks" onclick="openCity(event, 'Cliques')">Cliques</button>
			<button class="tablinks" onclick="openCity(event, 'Course Tracks')">CS Tracks</button>
			<button class="tablinks" onclick="openCity(event, 'Corr > 0.3')">Corr > 0.3</button>
			<button class="tablinks" onclick="openCity(event, 'Corr > 0.5')">Corr > 0.5</button>
			<button class="tablinks" onclick="openCity(event, 'Corr > 0.65')">Corr > 0.65</button>
		</div>

		<!-- Tab content -->
		<div id="GPAs" class="tabcontent">
			
.. raw:: html
		:file: gpaHistogram.html

.. raw:: html

		</div>

		<div id="Cliques" class="tabcontent">
			
.. raw:: html
		:file: cliqueHistogram.html

.. raw:: html

		</div>

		<div id="Course Tracks" class="tabcontent">
			
.. raw:: html
		:file: sankey2.html

.. raw:: html

		</div>

		<div id="Corr > 0.3" class="tabcontent">
			
.. raw:: html
		:file: gradeGraph30.html

.. raw:: html

		</div>

		<div id="Corr > 0.5" class="tabcontent">
			
.. raw:: html
		:file: gradeGraph50.html

.. raw:: html

		</div>

		<div id="Corr > 0.65" class="tabcontent">
			
.. raw:: html
		:file: gradeGraph65.html

.. raw:: html

		<script type="text/javascript">
            openCity(event, 'GPAs');
        </script>
		</div>
.. .. raw:: html

..     <div class="row">
..       <div class="column" style="background-color:#FFF;">
..         <h2>GPA Distribution</h2>
				
.. .. raw:: html
.. 		:file: gpaHistogram.html

.. .. raw:: html

..       </div>
..       <div class="column" style="background-color:#FFF;">
..         <h2>Clique Distribution</h2>

.. .. raw:: html
.. 		:file: cliqueHistogram.html

.. .. raw:: html

..       </div>
..     </div>

.. Grade Correlations 
.. ------------------

.. *correlation > 0.3, P-value < 0.05*


.. `________________________________________________________________`

.. .. raw:: html
.. 		:file: gradeGraph30.html
		
.. .. raw:: html

.. 		<br />
.. 		<br />

.. *correlation > 0.5, P-value < 0.05*


.. `________________________________________________________________`

.. .. raw:: html
.. 		:file: gradeGraph50.html	

.. .. raw:: html

.. 		<br />
.. 		<br />

.. *correlation > 0.65, P-value < 0.05*


.. `________________________________________________________________`

.. .. raw:: html
.. 		:file: gradeGraph65.html
