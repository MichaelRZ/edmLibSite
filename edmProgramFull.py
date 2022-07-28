import csv, sys, os, io
import PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
# from PyQt5.QtWebEngineWidgets import *
import pandas as pd
import numpy as np
import inspect

"""
Library created by Michael Riad Zaky mriadzaky@fordham.edu (2020), with Daniel Leeds, Gary Weiss, Mavis Zhang at Fordham 
University. Library free for use provided you cite https://github.com/MichaelRZ/EDMLib in any resulting publications.  
Library free for redistribution provided you retain the author attributions above.

The following packages are required for installation before use: numpy, pandas, csv, scipy, holoviews
"""
import time
import math
from scipy.stats.stats import pearsonr
import re
import networkx as nx
import itertools
import holoviews as hv
from holoviews import opts, dim
from bokeh.plotting import show, save, output_file
from bokeh.io import export_png
from bokeh.models import Title
numLibInstalled = True
try:
  import numexpr
except:
  numLibInstalled = False
  pass

pd.options.mode.chained_assignment = None 
hv.extension('bokeh')
def disable_logo(plot, element):
    plot.state.toolbar.logo = None
hv.plotting.bokeh.ElementPlot.hooks.append(disable_logo)

edmApplication = False

class gradeData:
  """Class for manipulating grade datasets.

    Attributes:
        df (:obj:`pandas.dataframe`): dataframe containing all grade data.
        sourceFile (:obj:`str`): Name of source .CSV file with grade data (optional).

  """
  ######## Variables #########
  df = None
  sourceFile = ""

  FINAL_GRADE_COLUMN = 'finalGrade'
  STUDENT_ID_COLUMN = 'studentID'
  FACULTY_ID_COLUMN = 'facultyID'
  CLASS_NUMBER_COLUMN = 'classNumber'
  CLASS_DEPT_COLUMN = 'classDept'
  TERM_COLUMN = 'term'
  CLASS_ID_COLUMN = 'classID'
  STUDENT_MAJOR_COLUMN = 'studentMajor'
  CLASS_CREDITS_COLUMN = 'classCredits'

  CLASS_ID_AND_TERM_COLUMN = 'courseIdAndTerm'
  CLASS_CODE_COLUMN = 'classCode'
  GPA_STDDEV_COLUMN = 'gpaStdDeviation'
  GPA_MEAN_COLUMN = 'gpaMean'
  NORMALIZATION_COLUMN = 'norm'

  ######## Methods #########
  def __init__(self,sourceFileOrDataFrame):
    """Class constructor, creates an instance of the class given a .CSV file or pandas dataframe.

    Used with gradeData('fileName.csv') or gradeData(dataFrameVariable).

    Args:
        sourceFileOrDataFrame (:obj:`object`): name of the .CSV file (extension included) in the same path or pandas dataframe variable. Dataframes are copied so as to not affect the original variable.

    """
    if type(sourceFileOrDataFrame).__name__ == 'str':
      self.sourceFile = sourceFileOrDataFrame
      self.df = pd.read_csv(self.sourceFile, dtype=str)

    elif type(sourceFileOrDataFrame).__name__ == 'DataFrame':
      if not edmApplication:
        self.df = sourceFileOrDataFrame.copy()
      else:
        self.df = sourceFileOrDataFrame

  def getColumn(self, column):
    """Returns a given column.

    Args:
        column (:obj:`str`): name of the column to return.

    Returns:
        :obj:`pandas.series`: column contained in pandas dataframe.

    """
    return self.df[column]

  def printColumn(self, column):
    """Prints the given column.

    Args:
        column (:obj:`str`): Column to print to console.
        
    """
    print(self.df[column])

  def defineWorkingColumns(self, finalGrade, studentID, term, classID = 'classID', classDept = 'classDept', classNumber = 'classNumber', studentMajor = 'studentMajor', classCredits = 'classCredits', facultyID = 'facultyID', classCode = 'classCode'):
    """Defines the column constants to target in the pandas dataframe for data manipulation. Required for proper functioning 
    of the library.

    Note:
        Either both :obj:`classDept` and :obj:`classNumber` variables need to be defined in the dataset's columns, or :obj:`classCode` needs to be defined for the library to function. The opposite variable(s) are then optional. :obj:`classDept` needs to be defined for major related functions to work.

    Args:
        finalGrade (:obj:`str`): Name of the column with grades given to a student in the respective class, grades are expected to be on a 1.0 - 4.0+ scale.
        studentID (:obj:`str`): Name of the column with student IDs, which does not need to follow any format.
        term (:obj:`str`): Name of the column with the term the class was taken, does not need to follow any format.
        classID (:obj:`str`): Number or string specific to a given class section, does not need to follow any format.
        classDept (:obj:`str`, optional): Name of the column stating the department of the class, e.g. 'Psych'. Defaults to 'classDept'. 
        classNumber (:obj:`str`, optional): Name of the column stating a number associated with the class, e.g. '1000' in 'Psych1000' or 'Intro to Psych'. Defaults to 'classNumber'.
        studentMajor (:obj:`str`, optional): Name of the column stating the major of the student. Optional, but required for functions involving student majors.
        classCredits (:obj:`str`, optional): Name of the column stating the number of credits a class is worth. Optional, but can be used to make student GPA calculations more accurate.
        facultyID (:obj:`str`, optional): Name of the column with faculty IDs, which does not need to follow any format. This is the faculty that taught the class. Optional, but required for instructor effectiveness functions.
        classCode (:obj:`str`, optional): Name of the column defining a class specific name, e.g. 'Psych1000'. Defaults to 'classCode'.

    """
    self.FINAL_GRADE_COLUMN = finalGrade
    self.STUDENT_ID_COLUMN = studentID
    self.FACULTY_ID_COLUMN = facultyID
    self.CLASS_ID_COLUMN = classID
    self.CLASS_DEPT_COLUMN = classDept
    self.CLASS_NUMBER_COLUMN = classNumber
    self.TERM_COLUMN = term
    self.CLASS_CODE_COLUMN = classCode
    self.STUDENT_MAJOR_COLUMN = studentMajor
    self.CLASS_CREDITS_COLUMN = classCredits

  def makeMissingValuesNanInColumn(self, column):
    if not self.__requiredColumnPresent(column):
      return
    self.df[column].replace(' ', np.nan, inplace=True)

  def removeNanInColumn(self, column):
    if not self.__requiredColumnPresent(column):
      return
    self.df.dropna(subset=[column], inplace=True)
    self.df.reset_index(inplace = True, drop=True)

  def dropMissingValuesInColumn(self, column):
    """Removes rows in the dataset which have missing data in the given column.

      Args:
        column (:obj:`str`): Column to check for missing values in.

    """
    if not self.__requiredColumnPresent(column):
      return
    self.makeMissingValuesNanInColumn(column)
    self.removeNanInColumn(column)
    

  def convertColumnToNumeric(self, column):
    if not self.__requiredColumnPresent(column):
      return
    self.df[column] = pd.to_numeric(self.df[column])

  def convertColumnToString(self, column):
    if not self.__requiredColumnPresent(column):
      return
    self.df.astype({column:str}, copy=False)

  def outputGpaDistribution(self, makeHistogram = False, fileName = 'gpaHistogram', graphTitle='GPA Distribution', minClasses = 36):
    """Prints to console an overview of student GPAs in increments of 0.1 between 1.0 and 4.0. Optionally, outputs a histogram as well.

    Args:
        makeHistogram (:obj:`bool`, optional): Whether or not to make a histogram graph. Default false.
        fileName (:obj:`str`): Name of histogram files to output. Default 'gpaHistogram'.
        graphTitle (:obj:`str`): Title to display on graph. Default 'GPA Distribution'.
        minClasses (:obj:`int`): Number of classes a student needs to have on record to count GPA. Default 36.

    """
    if not self.__requiredColumnPresent(self.FINAL_GRADE_COLUMN):
      return
    if not self.__requiredColumnPresent(self.STUDENT_ID_COLUMN):
      return
    self.df[self.FINAL_GRADE_COLUMN] = pd.to_numeric(self.df[self.FINAL_GRADE_COLUMN], errors='coerce')
    if self.CLASS_CREDITS_COLUMN in self.df.columns:
      temp = self.df.loc[:, [self.STUDENT_ID_COLUMN, self.CLASS_CREDITS_COLUMN, self.FINAL_GRADE_COLUMN]]
    else:
      temp = self.df.loc[:, [self.STUDENT_ID_COLUMN, self.FINAL_GRADE_COLUMN]]
    classCount = temp[self.STUDENT_ID_COLUMN].value_counts()
    temp = temp[temp[self.STUDENT_ID_COLUMN].isin(classCount[classCount >= minClasses].index)]
    print('Number of Students: ' + str(temp[self.STUDENT_ID_COLUMN].nunique()))
    if temp[self.STUDENT_ID_COLUMN].nunique() == 0:
      print('Error: no students meet the criteria given')
      return
    if self.CLASS_CREDITS_COLUMN in self.df.columns:
      temp[self.CLASS_CREDITS_COLUMN] = pd.to_numeric(temp[self.CLASS_CREDITS_COLUMN], errors='coerce')
      temp['classPoints'] = temp[self.CLASS_CREDITS_COLUMN] * temp[self.FINAL_GRADE_COLUMN]
      sums = temp.groupby(self.STUDENT_ID_COLUMN).sum()
      sums['gpa'] = sums['classPoints'] / sums[self.CLASS_CREDITS_COLUMN]
      gpas = sums['gpa'].tolist()
    else:
      gradeAverages = temp.groupby(self.STUDENT_ID_COLUMN).mean()
      gpas = gradeAverages[self.FINAL_GRADE_COLUMN].tolist()
    mean = sum(gpas) / len(gpas)
    grade = 4.0
    print(">= "+ str(grade) + ": " + str(len([x for x in gpas if x >= grade])))
    while grade != 1.0:
      lowerGrade = round(grade - 0.1, 1)
      print(str(lowerGrade) + " - " + str(grade) + ": " + str(len([x for x in gpas if x >= lowerGrade and x < grade])))
      grade = round(grade - 0.1, 1)
    print("< "+ str(grade) + ": " + str(len([x for x in gpas if x < grade])))
    print('mean: ' + str(mean))
    if makeHistogram:
      lowest = round(float('%.1f'%(min(gpas))), 1)
      highest = round(max(gpas), 1)
      frequencies, edges = np.histogram(gpas, int((highest - lowest) / 0.1), (lowest, highest))
      #print('Values: %s, Edges: %s' % (frequencies.shape[0], edges.shape[0]))
      histo = hv.Histogram((edges, frequencies))
      histo.opts(opts.Histogram(xlabel='Student GPA', ylabel='Frequency', title=graphTitle))
      subtitle= 'mean: ' + str(round(sum(gpas) / len(gpas), 2))+ ', n = ' + str(temp[self.STUDENT_ID_COLUMN].nunique())
      hv.output(size=125)
      graph = hv.render(histo)
      graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="10pt"), 'above')
      output_file(fileName + '.html', mode='inline')
      save(graph)
      show(graph)
      if not edmApplication:
        histo.opts(toolbar=None)
        graph = hv.render(histo)
        graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="10pt"), 'above')
        export_png(graph, filename=fileName + '.png')

  def termMapping(self, mapping):
    self.df['termOrder'] = self.df[self.TERM_COLUMN].map(mapping)

  def sankeyGraphByCourseTracksOneGroup(self, courseGroup, requiredCourses = None, graphTitle='Track Distribution', outputName = 'sankeyGraph', minEdgeValue = None):
    """Exports a sankey graph according to a given course track. Input is organized as an array of classes included in the track, and optionally a subgroup of classes required for a student to be counted in the graph can be designated as well.

    Args:
        courseGroup (:obj:`list`(:obj:`str`)): List of courses to make the sankey graph with. Minimum two courses.
        requiredCourses (:obj:`list`(:obj:`str`)): List of courses required for a student to count towards the graph. All courses in 'courseGroup' by default.
        graphTitle (:obj:`str`): Title that goes on the sankey graph. Defaults to 'Track Distribution'.
        outputName (:obj:`str`): Name of sankey files (.csv, .html) to output. Defaults to 'sankeyGraph'.
        minEdgeValue (:obj:`int`, optional): Minimum value for an edge to be included on the sankey graph. Defaults to `None`, or no minimum value needed.

    """
    print('Creating Sankey graph...')
    if len(courseGroup) < 2:
      print('Error: Minimum of two courses required in given course track.')
      return
    if requiredCourses:
      requiredCourses = [x for x in requiredCourses if x in courseGroup]
      if len(requiredCourses) == 0:
        requiredCourses = None
    if not requiredCourses:
      requiredCourses = courseGroup
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      return
    #The following line is a function to get number suffixes. I don't know how it works, but it does.
    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(math.floor(n/10)%10!=1)*(n%10<4)*n%10::4])
    firstGroup = self.df.loc[self.df[self.CLASS_CODE_COLUMN].isin(courseGroup)]
    relevantStudents = firstGroup[self.STUDENT_ID_COLUMN].unique()
    edges = {}
    def addEdge(first, second, count):
      firstNode = getattr(first, self.CLASS_CODE_COLUMN) + ' ' + ordinal(count)
      secondNode = getattr(second, self.CLASS_CODE_COLUMN) + ' ' + ordinal(count + 1)
      pair = (firstNode, secondNode)
      # print(pair)
      if pair in edges:
        edges[pair] += 1
      else:
        edges[pair] = 1
    outOf = len(relevantStudents)
    stNum = 0
    for student in relevantStudents:
      stNum += 1
      print('student ' + str(stNum) + '/' + str(outOf))
      count = 1
      studentClasses = self.df.loc[self.df[self.STUDENT_ID_COLUMN]==student]
      correctClasses = studentClasses.loc[studentClasses[self.CLASS_CODE_COLUMN].isin(courseGroup)]
      sortedClasses = correctClasses.sort_values(self.TERM_COLUMN)
      uniqueClasses = set(sortedClasses[self.CLASS_CODE_COLUMN].unique())
      if all(course in uniqueClasses for course in requiredCourses) and len(sortedClasses.index) > 1:
        first = None
        second = None
        lastTerm = None
        for row in sortedClasses.itertuples(index=False):
          if not lastTerm:
            lastTerm = getattr(row, self.TERM_COLUMN)
          if getattr(row, self.TERM_COLUMN) > lastTerm:
            lastTerm = getattr(row, self.TERM_COLUMN)
            count += 1
          nextTerm = None
          for row2 in sortedClasses.itertuples(index=False):
            if getattr(row2, self.TERM_COLUMN) > getattr(row, self.TERM_COLUMN):
              if not nextTerm:
                nextTerm = getattr(row2, self.TERM_COLUMN)
                addEdge(row, row2, count)
              elif getattr(row2, self.TERM_COLUMN) == nextTerm:
                addEdge(row, row2, count)
              else:
                break
      
    edgeList = []
    skippedEdges = {}
    if minEdgeValue:
      for key, value in edges.items():
        if value < minEdgeValue:
          if key[1] in skippedEdges:
            skippedEdges[key[1]] += value
          else:
            skippedEdges[key[1]] = value
    for key, value in edges.items():
      if minEdgeValue:
        if key[0] in skippedEdges:
          value -= skippedEdges[key[0]]
        if value < minEdgeValue:
          continue
      temp = [key[0], key[1], value]
      edgeList.append(temp)
    sankey = hv.Sankey(edgeList, ['From', 'To'])
    sankey.opts(width=600, height=400, node_padding=40, edge_color_index='From', color_index='index', title=graphTitle)
    graph = hv.render(sankey)
    output_file(outputName + '.html', mode='inline')
    save(graph)
    show(graph)

  def sankeyGraphByCourseTracks(self, courseGroups, graphTitle='Track Distribution', outputName = 'sankeyGraph', consecutive = True, minEdgeValue = None, termThreshold = None):
    """Exports a sankey graph according to a given course track. Input is organized in a jagged array, with the first array the first set of classes a student can take, the second set the second possible class a student can take, etc..

    Args:
        courseGroups (:obj:`list`(:obj:`list`)): List of course groups (also lists) to make the sankey graph with. Minimum two course groups.
        graphTitle (:obj:`str`): Title that goes on the sankey graph. Defaults to 'Track Distribution'.
        outputName (:obj:`str`): Name of sankey files (.csv, .html) to output. Defaults to 'sankeyGraph'.
        consecutive (:obj:`bool`): Whether or not students must complete the entire track consecutively, or start at a group other than what is designated. This mostly affects students who needed to retake a class. Defaults to :obj:`True` (students must complete track from beginning / as designated for data to be recorded).
        minEdgeValue (:obj:`int`, optional): Minimum value for an edge to be included on the sankey graph. Defaults to `None`, or no minimum value needed.
        termThreshold (:obj:`float`, optional): If defined, attempts to use the 'termOrder' column where terms are given a numbered order and a given maximum threshold for what counts as a "consecutive" term.

    """
    print('Creating Sankey graph...')
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      return
    classSet = list(set(itertools.chain.from_iterable(courseGroups)))
    #The following line is a function to get number suffixes. I don't know how it works, but it does.
    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(math.floor(n/10)%10!=1)*(n%10<4)*n%10::4])
    firstGroup = self.df.loc[np.in1d(self.df[self.CLASS_CODE_COLUMN], courseGroups[0])]
    relevantStudents = firstGroup[self.STUDENT_ID_COLUMN].unique()
    edges = {}
    def addEdge(first, second, count):
      if consecutive:
        if (first[self.CLASS_CODE_COLUMN] not in courseGroups[count - 1]) or (second[self.CLASS_CODE_COLUMN] not in courseGroups[count]):
          return
      firstNode = first[self.CLASS_CODE_COLUMN] + ' ' + ordinal(count)
      secondNode = second[self.CLASS_CODE_COLUMN] + ' ' + ordinal(count + 1)
      pair = (firstNode, secondNode)
      print(pair)
      if pair in edges:
        edges[pair] += 1
      else:
        edges[pair] = 1
    outOf = len(relevantStudents)
    stNum = 0
    for student in relevantStudents:
      stNum += 1
      print('student ' + str(stNum) + '/' + str(outOf))
      count = 0
      studentClasses = self.df.loc[self.df[self.STUDENT_ID_COLUMN]==student]
      correctClasses = studentClasses.loc[np.in1d(studentClasses[self.CLASS_CODE_COLUMN], classSet)]
      sortedClasses = correctClasses.sort_values(self.TERM_COLUMN)
      if 'termOrder' in self.df.columns:
        print('sorting')
        sortedClasses['termOrder'] = sortedClasses['termOrder'].astype(float)
        sortedClasses = sortedClasses.sort_values('termOrder')
      sortedClasses.reset_index(inplace = True)
      firstClass = False
      currentTerm = None
      i = 0
      j = 0
      rounded = lambda x: round(float(x),2)
      # print(sortedClasses[self.CLASS_CODE_COLUMN])
      # print(sortedClasses['termOrder'])
      while i < sortedClasses.index[-1]:
        if sortedClasses.iloc[i][self.CLASS_CODE_COLUMN] in courseGroups[count]:
          firstClass = True
          currentTerm = sortedClasses.iloc[i][self.TERM_COLUMN]
          if termThreshold:
            termNum = rounded(sortedClasses.iloc[i]['termOrder'])
          count += 1
          break
        i += 1
      if firstClass:
        if not termThreshold:
          while i < sortedClasses.index[-1]:
            if sortedClasses.iloc[i][self.TERM_COLUMN] != currentTerm:
              currentTerm = sortedClasses.iloc[i][self.TERM_COLUMN]
              count += 1
            if count >= len(courseGroups):
              break
            if sortedClasses.iloc[i][self.CLASS_CODE_COLUMN] not in courseGroups[count-1]:
              i += 1
              continue
            j = i + 1
            while j <= sortedClasses.index[-1]:
              if sortedClasses.iloc[j][self.TERM_COLUMN] == currentTerm:
                j += 1
                continue
              nextTerm = sortedClasses.iloc[j][self.TERM_COLUMN]
              break
            while j <= sortedClasses.index[-1] and sortedClasses.iloc[j][self.TERM_COLUMN] == nextTerm:
              if (sortedClasses.iloc[j][self.CLASS_CODE_COLUMN] in courseGroups[count]):
                addEdge(sortedClasses.iloc[i], sortedClasses.iloc[j], count)
              j += 1
            i += 1
        else:
          # print(sortedClasses)
          while i < sortedClasses.index[-1]:
            if rounded(sortedClasses.iloc[i]['termOrder']) != rounded(termNum):
              termNum = rounded(sortedClasses.iloc[i]['termOrder'])
              count += 1
            if count >= len(courseGroups):
              break
            if sortedClasses.iloc[i][self.CLASS_CODE_COLUMN] not in courseGroups[count-1]:
              i += 1
              continue
            j = i + 1
            while j <= sortedClasses.index[-1]:
              if rounded(sortedClasses.iloc[j]['termOrder']) == rounded(termNum):
                j += 1
                continue
              break
            while j <= sortedClasses.index[-1] and rounded(rounded(sortedClasses.iloc[j]['termOrder']) - termNum) <= rounded(termThreshold):
              if (sortedClasses.iloc[j][self.CLASS_CODE_COLUMN] in courseGroups[count]):
                addEdge(sortedClasses.iloc[i], sortedClasses.iloc[j], count)
              j += 1
            i += 1
        # for row in sortedClasses.index[1:]:
        #   current = sortedClasses.iloc[row]
        #   if lastClass[self.TERM_COLUMN] == current[self.TERM_COLUMN]:
        #     continue
        #   if current[self.CLASS_CODE_COLUMN] in courseGroups[count]:
        #     first = sortedClasses.iloc[row]
        #     second = sortedClasses.iloc[row+1]
        #     if first[self.TERM_COLUMN] != second[self.TERM_COLUMN]:
        #       lastClass = first   
        #       sameTerm = True
        #       count += 1
        #     if first[self.TERM_COLUMN] != second[self.TERM_COLUMN] and second[self.CLASS_CODE_COLUMN] in courseGroups[count]:
        #       addEdge(first, second, count)
        #       count += 1
        #       if count >= len(courseGroups) - 1:
        #         break
        #     elif sameTerm and second[self.CLASS_CODE_COLUMN] in courseGroups[count]:
        #       addEdge(lastClass, second, count+1)
        # lastTerm = None
        # for row in sortedClasses.index[:-1]:
        #   first = sortedClasses.iloc[row]

    edgeList = []
    skippedEdges = {}
    # print(edges)
    # print(minEdgeValue)
    if minEdgeValue:
      for key, value in edges.items():
        if value < minEdgeValue:
          if key[1] in skippedEdges:
            skippedEdges[key[1]] += value
          else:
            skippedEdges[key[1]] = value
    for key, value in edges.items():
      if minEdgeValue:
        if key[0] in skippedEdges:
          value -= skippedEdges[key[0]]
        if value < minEdgeValue:
          continue
      temp = [key[0], key[1], value]
      edgeList.append(temp)
    sankey = hv.Sankey(edgeList, ['From', 'To'])
    sankey.opts(width=600, height=400, node_padding=40, edge_color_index='From', color_index='index', title=graphTitle)
    graph = hv.render(sankey)
    output_file(outputName + '.html', mode='inline')
    save(graph)
    show(graph)

  def substituteSubStrInColumn(self, column, subString, substitute):
    """Replace a substring in a given column.

      Args:
        column (:obj:`str`): Column to replace substring in.
        subString (:obj:`str`): Substring to replace.
        substitute (:obj:`str`): Replacement of the substring.

    """
    self.convertColumnToString(column)
    self.df[column] = self.df[column].str.replace(subString, substitute)
    # print(self.df[column])

  def getUniqueIdentifiersForSectionsAcrossTerms(self):
    """Used internally. If a column 'classCode' is unavailable, a new column is made by combining 'classDept' and 
    'classNumber' columns. Also makes a new 'classIdAndTerm' column by combining the 'classID' and 'term' columns,
    to differentiate specific class sections in specific terms.

    """
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      if not self.__requiredColumnPresent(self.CLASS_DEPT_COLUMN):
        print("Note: Optionally, the 'classDept' column does not need to be defined if the class specific (e.g. 'Psych1000' or 'IntroToPsych') column 'classCode' is defined. This can be done with the 'defineWorkingColumns' function. This does, however, break department or major specific course functions.")
        return
      if not self.__requiredColumnPresent(self.CLASS_NUMBER_COLUMN):
        print("Note: Optionally, the 'classNumber' column does not need to be defined if the class specific (e.g. 'Psych1000' or 'IntroToPsych') column 'classCode' is defined. This can be done with the 'defineWorkingColumns' function.")
        return
      self.df[self.CLASS_CODE_COLUMN] = self.df[self.CLASS_DEPT_COLUMN].astype(str) + self.df[self.CLASS_NUMBER_COLUMN]
      self.df[self.CLASS_CODE_COLUMN] = self.df[self.CLASS_CODE_COLUMN].str.replace(" ","")
    if self.CLASS_ID_AND_TERM_COLUMN not in self.df.columns:
      if not self.__requiredColumnPresent(self.CLASS_ID_COLUMN):
        return
      if not self.__requiredColumnPresent(self.TERM_COLUMN):
        return
      self.df[self.CLASS_ID_AND_TERM_COLUMN] = self.df[self.CLASS_ID_COLUMN].astype(str) + self.df[self.TERM_COLUMN]
      self.df[self.CLASS_ID_AND_TERM_COLUMN] = self.df[self.CLASS_ID_AND_TERM_COLUMN].str.replace(" ","")

  def instructorRanks(self, firstClass, secondClass, fileName = 'instructorRanking', minStudents = 1):
    """Create a table of instructors and their calculated benefit to students based on a class they taught and future performance in a given class taken later. Exports a CSV file and returns a pandas dataframe.

      Args:
        firstClass (:obj:`str`): Class to look at instructors / their students from.
        secondClass (:obj:`str`): Class to look at future performance of students who had relevant professors from the first class.
        fileName (:obj:`str`, optional): Name of CSV file to save. Set to 'instructorRanking' by default.
        minStudents (:obj:`int`, optional): Minimum number of students to get data from for an instructor to be included in the calculation. Set to 1 by default.

      Returns:
        :obj:`pandas.dataframe`: Pandas dataframe with columns indicating the instructor, the normalized benefit to students, the grade point benefit to students, and the number of students used to calculate for that instructor.

    """
    if not self.__requiredColumnPresent(self.FACULTY_ID_COLUMN):
      return
    if self.NORMALIZATION_COLUMN not in self.df.columns:
      self.getNormalizationColumn()
    if not self.__requiredColumnPresent(self.NORMALIZATION_COLUMN):
      return
    firstClassEntries = self.df.loc[self.df[self.CLASS_CODE_COLUMN] == firstClass]
    secondClassEntries = self.df.loc[self.df[self.CLASS_CODE_COLUMN] == secondClass]

    instructors = firstClassEntries[self.FACULTY_ID_COLUMN].unique()
    instructorRank = {}
    instructorNorm = {}
    instructorStudCount = {}

    for instructor in instructors:
      instructorStudCount[instructor] = 0
      tookInstructor = firstClassEntries.loc[firstClassEntries[self.FACULTY_ID_COLUMN] == instructor]
      studentsWithInstructor = tookInstructor[self.STUDENT_ID_COLUMN].unique()
      secondClassWithPastInstructor = secondClassEntries[self.STUDENT_ID_COLUMN].isin(studentsWithInstructor)
      if any(secondClassWithPastInstructor):
        instructorStudCount[instructor] = sum(secondClassWithPastInstructor)
        entriesWithPastInstructor = secondClassEntries.loc[secondClassWithPastInstructor]
        entriesWithoutPastInstructor = secondClassEntries.loc[~secondClassWithPastInstructor]
        AverageGradeWithInstructor = entriesWithPastInstructor[self.FINAL_GRADE_COLUMN].mean()
        AverageGradeWithoutInstructor = entriesWithoutPastInstructor[self.FINAL_GRADE_COLUMN].mean()
        stdDev = secondClassEntries[self.FINAL_GRADE_COLUMN].std()
        instructorRank[instructor] = (AverageGradeWithInstructor - AverageGradeWithoutInstructor) / stdDev
        instructorNorm[instructor] = entriesWithPastInstructor[self.NORMALIZATION_COLUMN].mean() - entriesWithoutPastInstructor[self.NORMALIZATION_COLUMN].mean()
    
    sortedRanks = sorted(instructorNorm.items(), key=lambda x: x[1], reverse=True)
    rankDf = pd.DataFrame(sortedRanks, columns=['Instructor(' + firstClass + ')', 'NormBenefit(' + secondClass + ')'])
    rankDf['GradeBenefit('+secondClass+')'] = rankDf['Instructor('+firstClass+')'].apply(lambda inst: instructorRank[inst])
    rankDf['#students'] = rankDf['Instructor('+firstClass+')'].apply(lambda inst: instructorStudCount[inst])    
    rankDf = rankDf.loc[rankDf['#students'] >= minStudents]
    if not fileName.endswith('.csv'):
      fileName = "".join((fileName, '.csv'))
    rankDf.to_csv(fileName, index=False)
    return rankDf

  def gradePredict(self, priorGrades, futureClasses, method='nearest', excludedStudents = None, normalized = False):
    """Predicts grades given a student's past grades and classes to predict for. Still being developed.

      Args:
        priorGrades (:obj:`dict`(:obj:`str` : :obj:`float`)): Dictionary of past courses and the respective grade recieved.
        futureClasses (:obj:`list`(:obj:`str`)): List of courses to predict grades for.
        method (:obj:`str`, optional): Method to use to predict grades. Current methods include 'nearest' which gives the grade recieved by the most similar student on record and 'nearestThree' which gives the grade closest to the mean of the grades recieved by the nearest three students on record. Set to 'nearest' by default.
        excludedStudents (:obj:`list`, optional): List of students to exclude when making calculation. Used for accuracy testing purposes. Set to :obj:`None` by default.
        normalized (:obj:`bool`, optional): Whether or not normalized grades are given as input. Used for accuracy testing purposes and should generally be set to :obj:`False`. Set to :obj:`False` by default.

      Returns:
        :obj:`dict`(:obj:`str` : :obj:`float`): Dictionary of grade predictions, where the key is the class and the value is the grade predicted.

    """
    if not self.__requiredColumnPresent(self.STUDENT_ID_COLUMN):
      return
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
      if not self.__requiredColumnPresent(self.CLASS_CODE_COLUMN):
        return
    if normalized:
      if self.NORMALIZATION_COLUMN not in self.df.columns:
        self.getNormalizationColumn()
        if not self.__requiredColumnPresent(self.NORMALIZATION_COLUMN):
          return
    else:
      self.dropMissingValuesInColumn(self.FINAL_GRADE_COLUMN)
      self.convertColumnToNumeric(self.FINAL_GRADE_COLUMN)
    print('Mode set to ' + method)
    relevantClasses = list(priorGrades.keys()) + futureClasses
    relevantEntries = self.df.loc[self.df[self.CLASS_CODE_COLUMN].isin(relevantClasses)]
    if excludedStudents:
      relevantEntries = relevantEntries.loc[~(relevantEntries[self.STUDENT_ID_COLUMN].isin(excludedStudents))]
    studentDict = relevantEntries.groupby(self.STUDENT_ID_COLUMN)
    prediction = {}
    if normalized:
      priorGradeComparison = lambda g: 10 - abs(priorGrades[g[self.CLASS_CODE_COLUMN]] - g[self.NORMALIZATION_COLUMN])
      def priorGradeDistance(grades, prior):
        A = grades[self.NORMALIZATION_COLUMN].values
        B = np.array([prior[x] for x in grades[self.CLASS_CODE_COLUMN].values])
        return np.linalg.norm(A - B)
    else:
      priorGradeComparison = lambda g: 1 - abs(priorGrades[g[self.CLASS_CODE_COLUMN]] - g[self.FINAL_GRADE_COLUMN])
      def priorGradeDistance(grades, prior):
        A = grades[self.FINAL_GRADE_COLUMN].values
        B = np.array([prior[x] for x in grades[self.CLASS_CODE_COLUMN].values])
        return np.linalg.norm(A - B)
    validGrades = self.df[self.FINAL_GRADE_COLUMN].unique()
    validGrades.sort()
    def gradePachinko(grades):
      meanGrade = sum(grades) / len(grades)
      idx = validGrades.searchsorted(meanGrade)
      idx = np.clip(idx, 1, len(validGrades)-1)
      left = validGrades[idx-1]
      right = validGrades[idx]
      idx -= meanGrade - left < right - meanGrade
      return validGrades[idx]

    for futureClass in futureClasses:
      print('Calculating for ' + futureClass + '...')
      futureClassEntries = relevantEntries.loc[relevantEntries[self.CLASS_CODE_COLUMN] == futureClass]
      relevantStudents = futureClassEntries[self.STUDENT_ID_COLUMN].unique()
      internalScore = {}
      numCommonClasses = {}
      # counter = 0
      # outOf = len(relevantStudents)
      for student in relevantStudents:
        # counter += 1
        # print(str(counter) + '/' + str(outOf) + ' students')
        commonClass = studentDict.get_group(student)[self.CLASS_CODE_COLUMN].isin(priorGrades.keys())
        numCommonClasses[student] = sum(commonClass)
        if numCommonClasses[student] > 0:
          relevantClasses = studentDict.get_group(student).loc[commonClass]
          applied = relevantClasses.apply(priorGradeComparison, axis=1)
          internalScore[student] = np.sum(applied.values)
          # internalScore[student] = priorGradeDistance(relevantClasses, priorGrades)
      if method == 'nearest' or len(internalScore) < 2:
        mostRelevant = max(internalScore, key=internalScore.get)
        # mostRelevant = min(internalScore, key=internalScore.get)
        # print(studentDict.get_group(mostRelevant)[[self.CLASS_CODE_COLUMN, self.FINAL_GRADE_COLUMN]])
        prediction[futureClass] = futureClassEntries.loc[futureClassEntries[self.STUDENT_ID_COLUMN] == mostRelevant].iloc[0][self.FINAL_GRADE_COLUMN]
      elif method == 'nearestThree':
        mostRelevant = sorted(internalScore, key=internalScore.get, reverse=True)[:3]
        # mostRelevant = sorted(internalScore, key=internalScore.get, reverse=False)[:3]
        relevantScores = [futureClassEntries.loc[futureClassEntries[self.STUDENT_ID_COLUMN] == stud].iloc[0][self.FINAL_GRADE_COLUMN] for stud in mostRelevant]
        prediction[futureClass] = gradePachinko(relevantScores)


    print(str(prediction)[1:-1])
    return prediction

  def getDictOfStudentMajors(self):
    """Returns a dictionary of students and their latest respective declared majors. Student ID, Student Major, and Term columns are required.

      Args: 
        N/A

      Returns:
        :obj:`dict`(:obj:`str` : :obj:`str`): Dictionary of students and their latest respective declared majors.

    """
    check = [self.__requiredColumnPresent(self.STUDENT_ID_COLUMN), self.__requiredColumnPresent(self.STUDENT_MAJOR_COLUMN), self.__requiredColumnPresent(self.TERM_COLUMN)]
    if not all(check):
      return
    lastEntries = self.df.sort_values(self.TERM_COLUMN)
    lastEntries.drop_duplicates(self.STUDENT_ID_COLUMN, keep='last', inplace=True)
    return pd.Series(lastEntries[self.STUDENT_MAJOR_COLUMN].values,index=df[self.STUDENT_ID_COLUMN]).to_dict()

    

  def instructorRanksAllClasses(self, fileName = 'completeInstructorRanks', minStudents = 20, directionality = 0.8):
    """Create a table of instructors and their calculated benefit to students based on all classes they taught and future performance in all classes taken later. Exports a CSV file and returns a pandas dataframe.

      Args:
        fileName (:obj:`str`, optional): Name of CSV file to save. Set to 'completeInstructorRanks' by default.
        minStudents (:obj:`int`, optional): Minimum number of students to get data from for an instructor's entry to be included in the calculation. Set to 1 by default.

      Returns:
        :obj:`pandas.dataframe`: Pandas dataframe with columns indicating the instructor, the class taken, the future class, the normalized benefit to students, the grade point benefit to students, the number of students used to calculate for that instructor / class combination, as well as the number of students on the opposite side of that calculation (students in future class who did not take that instructor before).

    """
    if not self.__requiredColumnPresent(self.FACULTY_ID_COLUMN):
      return
    if self.NORMALIZATION_COLUMN not in self.df.columns:
      self.getNormalizationColumn()
    if not self.__requiredColumnPresent(self.NORMALIZATION_COLUMN):
      return
    if directionality > 1.0 or directionality < 0.5:
      print('Error: directionality out of bounds (must be between 0.5 to 1, not '+ str(directionality) +').')
    
    rowList = []
    def processPair(classOne, classTwo, df):
      firstClass = df[self.CLASS_CODE_COLUMN] == classOne
      firstClassEntries = df.loc[firstClass]
      secondClassEntries = df.loc[~firstClass]
      instructorDict = firstClassEntries[self.FACULTY_ID_COLUMN].value_counts().to_dict()
      instructors = {key:val for key, val in instructorDict.items() if val >= minStudents}
      if not instructors:
        return
      for instructor, count in instructors.items():
        tookInstructor = firstClassEntries.loc[firstClassEntries[self.FACULTY_ID_COLUMN] == instructor]
        studentsWithInstructor = tookInstructor[self.STUDENT_ID_COLUMN].unique()
        secondClassWithPastInstructor = secondClassEntries[self.STUDENT_ID_COLUMN].isin(studentsWithInstructor)
        nonStudents = len(secondClassWithPastInstructor.index) - sum(secondClassWithPastInstructor)
        if nonStudents > 0:
          entriesWithPastInstructor = secondClassEntries.loc[secondClassWithPastInstructor]
          entriesWithoutPastInstructor = secondClassEntries.loc[~secondClassWithPastInstructor]
          AverageGradeWithInstructor = entriesWithPastInstructor[self.FINAL_GRADE_COLUMN].mean()
          AverageGradeWithoutInstructor = entriesWithoutPastInstructor[self.FINAL_GRADE_COLUMN].mean()
          stdDev = secondClassEntries[self.FINAL_GRADE_COLUMN].std()
          rowDict = {}
          rowDict['Instructor'] = instructor
          rowDict['courseTaught'] = classOne
          rowDict['futureCourse'] = classTwo
          rowDict['normBenefit'] = entriesWithPastInstructor[self.NORMALIZATION_COLUMN].mean() - entriesWithoutPastInstructor[self.NORMALIZATION_COLUMN].mean()
          rowDict['gradeBenefit'] = (AverageGradeWithInstructor - AverageGradeWithoutInstructor) / stdDev
          rowDict['#students'] = count
          rowDict['#nonStudents'] = nonStudents
          rowList.append(rowDict)

    classes = self.df[self.CLASS_CODE_COLUMN].unique().tolist()
    numClasses = len(classes)
    grouped = self.df.groupby(self.CLASS_CODE_COLUMN)
    for i in range(numClasses - 1):
      print('class ' + str(i+1) + '/' + str(numClasses))
      classOne = classes[i]
      start_time = time.time()
      for j in range(i + 1, numClasses):
        classTwo = classes[j]
        twoClasses = [classOne, classTwo]
        oneDf = grouped.get_group(classOne)
        twoDf = grouped.get_group(classTwo)
        studentInClass = np.intersect1d(oneDf[self.STUDENT_ID_COLUMN].values,twoDf[self.STUDENT_ID_COLUMN].values)
        if len(studentInClass) >= minStudents:
          combinedEntries = pd.concat([oneDf, twoDf], ignore_index=True)
          relevantEntries = combinedEntries.loc[combinedEntries[self.STUDENT_ID_COLUMN].isin(studentInClass)]
          relevantEntries.sort_values(self.TERM_COLUMN, inplace = True)
          firstEntries = relevantEntries[[self.STUDENT_ID_COLUMN, self.CLASS_CODE_COLUMN]].drop_duplicates(self.STUDENT_ID_COLUMN)
          classOneFirstCount = sum(firstEntries[self.CLASS_CODE_COLUMN] == classOne)
          directionOne = classOneFirstCount / (len(firstEntries.index))
          if directionOne >= directionality:
            processPair(classOne, classTwo, relevantEntries)
          if (1.0 - directionOne) >= directionality:
            processPair(classTwo, classOne, relevantEntries)
      # print('outerEnd: ' + str(time.time() - start_time))      
    
    completeDf = pd.DataFrame(rowList, columns=['Instructor','courseTaught','futureCourse','normBenefit','gradeBenefit','#students', '#nonStudents'])
    completeDf.sort_values(by=['futureCourse','courseTaught','Instructor'])
    completeDf['Instructor'].replace(' ', np.nan, inplace=True)
    completeDf.dropna(subset=['Instructor'], inplace=True)
    completeDf.reset_index(inplace = True, drop=True)
    if not fileName.endswith('.csv'):
      fileName = "".join((fileName, '.csv'))
    completeDf.to_csv(fileName, index=False)
    return completeDf

  def getCorrelationsWithMinNSharedStudents(self, nSharedStudents = 20, directed = False, classDetails = False):
    """Returns a pandas dataframe with correlations between all available classes based on grades, after normalization.

    Args:
        nSharedStudents (:obj:`int`, optional): Minimum number of shared students a pair of classes must have to compute a correlation. Defaults to 20.
        directed (:obj:`bool`, optional): Whether or not to include data specific to students who took class A before B, vice versa, and concurrently. Defaults to 'False'.
        classDetails (:obj:`bool`, optional): Whether or not to include means of student grades, normalized grades, and standard deviations used. Defaults to 'False'.

    Returns:
        :obj:`pandas.dataframe`: Pandas dataframe with at least columns "course1", "course2", "corr", "P-value", and "#students", which store class names, their correlation coefficient (0 least to 1 most), the P-value of this calculation, and the number of students shared between these two classes.

    """
    print("Getting correlations...")
    start_time = time.time()
    if self.NORMALIZATION_COLUMN not in self.df.columns:
      self.getNormalizationColumn()
    if not self.__requiredColumnPresent(self.NORMALIZATION_COLUMN):
      return
    if not self.__requiredColumnPresent(self.STUDENT_ID_COLUMN):
      return
    if not self.__requiredColumnPresent(self.STUDENT_ID_COLUMN):
      return
    if directed:
      if not self.__requiredColumnPresent(self.TERM_COLUMN):
        return
      self.df[self.TERM_COLUMN] = pd.to_numeric(self.df[self.TERM_COLUMN],errors='ignore')

    def corrAlg(a, b): 
      norms = a.loc[a[self.STUDENT_ID_COLUMN].isin(b[self.STUDENT_ID_COLUMN].values)]
      norms = norms.dropna(subset=[self.NORMALIZATION_COLUMN])
      if len(norms) >= nSharedStudents:
        norms.set_index(self.STUDENT_ID_COLUMN, inplace=True)
        norms.sort_index(inplace=True)
        norms2 = b.loc[b[self.STUDENT_ID_COLUMN].isin(a[self.STUDENT_ID_COLUMN].values)]
        norms2.set_index(self.STUDENT_ID_COLUMN, inplace=True)
        norms2.sort_index(inplace=True)
        corr, Pvalue = pearsonr(norms[self.NORMALIZATION_COLUMN], norms2[self.NORMALIZATION_COLUMN])
        return [corr, Pvalue, len(norms.index)]
      else:
        return [math.nan, math.nan, math.nan]
    def corrAlgDirected(a, b): 
      norms = a.loc[a[self.STUDENT_ID_COLUMN].isin(b[self.STUDENT_ID_COLUMN].values)]
      norms = norms.dropna(subset=[self.NORMALIZATION_COLUMN])
      if len(norms) < nSharedStudents:
        return [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, 
                math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, 
                math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan]
      norms.set_index(self.STUDENT_ID_COLUMN, inplace=True)
      norms.sort_index(inplace=True)
      aNorms = norms[self.NORMALIZATION_COLUMN]
      norms2 = b.loc[b[self.STUDENT_ID_COLUMN].isin(a[self.STUDENT_ID_COLUMN].values)]
      norms2.set_index(self.STUDENT_ID_COLUMN, inplace=True)
      norms2.sort_index(inplace=True)
      if np.issubdtype(norms[self.TERM_COLUMN].dtype, np.number) and np.issubdtype(norms2[self.TERM_COLUMN].dtype, np.number) and numLibInstalled:
        n = norms[self.TERM_COLUMN].values
        m = norms2[self.TERM_COLUMN].values
        less = numexpr.evaluate('(n < m)')
        more = numexpr.evaluate('(n > m)')
        same = numexpr.evaluate('(n == m)')
        concurrentA = norms.loc[same]
        concurrentB = norms2.loc[same]
      else:
        less = norms[self.TERM_COLUMN].values < norms2[self.TERM_COLUMN].values
        more = norms[self.TERM_COLUMN].values > norms2[self.TERM_COLUMN].values
        concurrentA = norms.loc[(~less) & (~more)]
        concurrentB = norms2.loc[(~less) & (~more)]
      aToBA = norms.loc[less]
      bToAA = norms.loc[more]
      aToBB = norms2.loc[less]
      bToAB = norms2.loc[more]
      bNorms = norms2[self.NORMALIZATION_COLUMN].dropna()
      abBNorms = aToBB[self.NORMALIZATION_COLUMN].dropna()
      baBNorms = bToAB[self.NORMALIZATION_COLUMN].dropna()
      concBNorms = concurrentB[self.NORMALIZATION_COLUMN].dropna()
      abANorms = aToBA[self.NORMALIZATION_COLUMN].dropna()
      baANorms = bToAA[self.NORMALIZATION_COLUMN].dropna()
      concANorms = concurrentA[self.NORMALIZATION_COLUMN].dropna()
      if classDetails:
        abAMean = aToBA[self.FINAL_GRADE_COLUMN].mean()
        abANormMean = aToBA[self.NORMALIZATION_COLUMN].mean()
        baAMean = bToAA[self.FINAL_GRADE_COLUMN].mean()
        baANormMean = bToAA[self.NORMALIZATION_COLUMN].mean()
        concAMean = concurrentA[self.FINAL_GRADE_COLUMN].mean()
        concANormMean = concurrentA[self.NORMALIZATION_COLUMN].mean()
        abBMean = aToBB[self.FINAL_GRADE_COLUMN].mean()
        abBNormMean = aToBB[self.NORMALIZATION_COLUMN].mean()
        baBMean = bToAB[self.FINAL_GRADE_COLUMN].mean()
        baBNormMean = bToAB[self.NORMALIZATION_COLUMN].mean()
        concBMean = concurrentB[self.FINAL_GRADE_COLUMN].mean()
        concBNormMean = concurrentB[self.NORMALIZATION_COLUMN].mean()

      corr, Pvalue = pearsonr(bNorms, aNorms)
      corr1, Pvalue1 = math.nan, math.nan
      corr2, Pvalue2 = math.nan, math.nan
      corr3, Pvalue3 = math.nan, math.nan
      if len(abANorms) >= 2:
        corr1, Pvalue1 = pearsonr(abBNorms,abANorms)
      if len(baANorms) >= 2:
        corr2, Pvalue2 = pearsonr(baBNorms,baANorms)
      if len(concANorms) >= 2:
        corr3, Pvalue3 = pearsonr(concBNorms,concANorms)
      if not classDetails:
        return [corr, Pvalue, len(aNorms), corr1, Pvalue1, len(abANorms), corr2, Pvalue2, 
        len(baANorms), corr3, Pvalue3, len(concANorms)]
      else:
        return [corr, Pvalue, len(aNorms), corr1, Pvalue1, len(abANorms), corr2, Pvalue2, len(baANorms), 
        corr3, Pvalue3, len(concANorms), abAMean, abANormMean, abBMean, abBNormMean, baAMean, baANormMean, 
        baBMean, baBNormMean, concAMean, concANormMean, concBMean, concBNormMean]

    print("Getting classes...")
    classes = self.getListOfClassCodes()
    d={}
    print("Organizing classes...")
    if not self._gradeData__requiredColumnPresent(self.CLASS_CODE_COLUMN):
      return
    
    if classDetails:
      rawGrades = {}
      normalizedGrades = {}
      stdDevGrade = {}
      self.convertColumnToNumeric(self.FINAL_GRADE_COLUMN)
      self.convertColumnToNumeric(self.NORMALIZATION_COLUMN)
      self.convertColumnToNumeric(self.GPA_STDDEV_COLUMN)
      
    for n, group in self.df.groupby(self.CLASS_CODE_COLUMN):
      d["df{0}".format(n)] = group
      d["df{0}".format(n)] = d["df{0}".format(n)].drop_duplicates(subset="SID", keep=False)
      # d["df{0}".format(n)].sort_values(by=[self.STUDENT_ID_COLUMN], inplace=True)
      if classDetails:
        raw = d["df{0}".format(n)][self.FINAL_GRADE_COLUMN].tolist()
        rawGrades[n] = str(sum(raw) / len(raw))
        # normalized = d["df{0}".format(n)][self.NORMALIZATION_COLUMN].tolist()
        # normalizedGrades[n] = str(sum(normalized) / len(normalized))
        stdDevGrade[n] = str(d["df{0}".format(n)][self.GPA_STDDEV_COLUMN].iloc[0])
        # print(rawGrades[n])
        # print(normalizedGrades[n])
        # print(stdDevGrade[n])

    f = []
    classCount = 0
    totalClasses = len(classes)
    print("Sorting classes...")
    classes.sort()
    classesProcessed = set()
    for n in classes:
      classCount = classCount + 1
      print("class " + str(classCount) + "/" + str(totalClasses))
      tim = time.time()
      for m in classes:
        if m not in classesProcessed:
          if not directed:
            classesProcessed.add(n)
            result = corrAlg(d["df{0}".format(n)],d["df{0}".format(m)])
            r, p, c = result[0], result[1], result[2]
            if not math.isnan(r):
              f.append((n, m, r, p, c))
              if n != m:
                f.append((m, n, r, p, c))
          else:
            classesProcessed.add(n)
            result = corrAlgDirected(d["df{0}".format(n)],d["df{0}".format(m)])
            r, p, c, r1, p1, c1, r2, p2, c2, r3, p3, c3 = result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8], result[9], result[10], result[11]
            
            if not math.isnan(r):
              if classDetails:
                abA, abANorm, abB, abBNorm, baA, baANorm, baB, baBNorm, concA, concANorm, concB, concBNorm = result[12], result[13], result[14], result[15], result[16], result[17], result[18], result[19], result[20], result[21], result[22], result[23]
                # print(n + " " + m + " " + str(r))
                f.append((n, m, r, p, c, r1, p1, c1, abA, abANorm, abB, abBNorm, r2, p2, c2, baB, baBNorm, baA, baANorm, r3, p3, c3, concA, concANorm, concB, concBNorm))
                if n != m:
                  f.append((m, n, r, p, c, r2, p2, c2, baB, baBNorm, baA, baANorm, r1, p1, c1, abA, abANorm, abB, abBNorm, r3, p3, c3, concB, concBNorm, concA, concANorm))
              else:
                f.append((n, m, r, p, c, r1, p1, c1, r2, p2, c2, r3, p3, c3))
                if n != m:
                  f.append((m, n, r, p, c, r2, p2, c2, r1, p1, c1, r3, p3, c3))
      classesProcessed.add(n)
      print(str(time.time() - tim))
    f[:] = [x for x in f if isinstance(x[0], str)]
    f.sort(key = lambda x: x[1])
    f.sort(key = lambda x: x[0])
    if not directed:
      normoutput = pd.DataFrame(f, columns=('course1', 'course2', 'corr', 'P-value', '#students'))
    else:
      if not classDetails:
        normoutput = pd.DataFrame(f, columns=('course1', 'course2', 'corr', 'P-value', '#students', 'corrCourse1->2', 'P-valueCrs1->2','#studentsCrs1->2', 'corrCourse2->1', 'P-valueCrs2->1','#studentsCrs2->1', 'corrCoursesConcurrent', 'P-valueCrsConcurrent','#studentsCrsConcurrent'))
      else:
        normoutput = pd.DataFrame(f, columns=('course1', 'course2', 'corr', 'P-value', '#students', 'corrCourse1->2', 'P-valueCrs1->2','#studentsCrs1->2', 'Av.GradeCrs1->2(crs1)', 'Av.NormCrs1->2(crs1)', 'Av.GradeCrs1->2(crs2)','Av.NormCrs1->2(crs2)','corrCourse2->1', 'P-valueCrs2->1','#studentsCrs2->1', 'Av.GradeCrs2->1(crs2)','Av.NormCrs2->1(crs2)', 'Av.GradeCrs2->1(crs1)', 'Av.NormCrs2->1(crs1)', 'corrCoursesConcurrent', 'P-valueCrsConcurrent','#studentsCrsConcurrent','Av.GradeConcurrent(crs1)','Av.NormConcurrent(crs1)','Av.GradeConcurrent(crs2)','Av.NormConcurrent(crs2)'))
    if classDetails:
      # rawGrades = {}
      # normalizedGrades = {}
      # stdDevGrade = {}
      normoutput['course1GradeMean'] = normoutput['course1'].apply(lambda x: rawGrades[x])
      # normoutput['course1NormalizedMean'] = normoutput['course1'].apply(lambda x: normalizedGrades[x])
      normoutput['course2GradeMean'] = normoutput['course2'].apply(lambda x: rawGrades[x])
      # normoutput['course2NormalizedMean'] = normoutput['course2'].apply(lambda x: normalizedGrades[x])
      normoutput['course1StdDev'] = normoutput['course1'].apply(lambda x: stdDevGrade[x])
      normoutput['course2StdDev'] = normoutput['course2'].apply(lambda x: stdDevGrade[x])
      normoutput['(Av.GradeCrs1->2(crs1)) - (Av.GradeCrs2->1(crs1))'] = normoutput['Av.GradeCrs1->2(crs1)'] - normoutput['Av.GradeCrs2->1(crs1)']
      normoutput['(Av.GradeCrs1->2(crs2)) - (Av.GradeCrs2->1(crs2))'] = normoutput['Av.GradeCrs1->2(crs2)'] - normoutput['Av.GradeCrs2->1(crs2)']
      normoutput['(Av.NormCrs1->2(crs1)) - (Av.NormCrs2->1(crs1))'] = normoutput['Av.NormCrs1->2(crs1)'] - normoutput['Av.NormCrs2->1(crs1)']
      normoutput['(Av.NormCrs1->2(crs2)) - (Av.NormCrs2->1(crs2))'] = normoutput['Av.NormCrs1->2(crs2)'] - normoutput['Av.NormCrs2->1(crs2)']
      normoutput['(Av.GradeCrs1->2(crs1)) - (Av.GradeConcurrent(crs1))'] = normoutput['Av.GradeCrs1->2(crs1)'] - normoutput['Av.GradeConcurrent(crs1)']
      normoutput['(Av.GradeCrs1->2(crs2)) - (Av.GradeConcurrent(crs2))'] = normoutput['Av.GradeCrs1->2(crs2)'] - normoutput['Av.GradeConcurrent(crs2)']
      normoutput['(Av.GradeCrs2->1(crs1)) - (Av.GradeConcurrent(crs1))'] = normoutput['Av.GradeCrs2->1(crs1)'] - normoutput['Av.GradeConcurrent(crs1)']
      normoutput['(Av.GradeCrs2->1(crs2)) - (Av.GradeConcurrent(crs2))'] = normoutput['Av.GradeCrs2->1(crs2)'] - normoutput['Av.GradeConcurrent(crs2)']
      normoutput['(Av.NormCrs1->2(crs1)) - (Av.NormConcurrent(crs1))'] = normoutput['Av.NormCrs1->2(crs1)'] - normoutput['Av.NormConcurrent(crs1)']
      normoutput['(Av.NormCrs1->2(crs2)) - (Av.NormConcurrent(crs2))'] = normoutput['Av.NormCrs1->2(crs2)'] - normoutput['Av.NormConcurrent(crs2)']
      normoutput['(Av.NormCrs2->1(crs1)) - (Av.NormConcurrent(crs1))'] = normoutput['Av.NormCrs2->1(crs1)'] - normoutput['Av.NormConcurrent(crs1)']
      normoutput['(Av.NormCrs2->1(crs2)) - (Av.NormConcurrent(crs2))'] = normoutput['Av.NormCrs2->1(crs2)'] - normoutput['Av.NormConcurrent(crs2)']
      normoutput['(#studentsCrs1->2) - (#studentsCrs2->1)'] = normoutput['#studentsCrs1->2'] - normoutput['#studentsCrs2->1']
      normoutput['(#studentsCrs1->2) - (#studentsCrsConcurrent)'] = normoutput['#studentsCrs1->2'] - normoutput['#studentsCrsConcurrent']
      normoutput['(#studentsCrs1->2) / (Total # students)'] = normoutput['#studentsCrs1->2'] / normoutput['#students']
      normoutput['(#studentsCrs2->1) / (Total # students)'] = normoutput['#studentsCrs2->1'] / normoutput['#students']
      normoutput['(#studentsCrsConcurrent) / (Total # students)'] = normoutput['#studentsCrsConcurrent'] / normoutput['#students']

    # normoutput = normoutput.dropna(how='all')
    print(str((totalClasses ** 2) - len(normoutput.index)) + ' class correlations dropped out of ' 
    + str(totalClasses ** 2) + ' from ' + str(nSharedStudents) + ' shared student threshold.')
    print(str(len(normoutput.index)) + ' correlations calculated. ' + str(time.time() - start_time) + ' seconds.')
    return normoutput

  def exportCorrelationsWithMinNSharedStudents(self, filename = 'CorrelationOutput_EDMLIB.csv', nStudents = 20, directedCorr = False, detailed = False):
    """Exports CSV file with all correlations between classes with the given minimum number of shared students. File format has columns 'course1', 'course2', 'corr', 'P-value', '#students'.

    Args:
        fileName (:obj:`str`, optional): Name of CSV to output. Default 'CorrelationOutput_EDMLIB.csv'.
        nStudents (:obj:`int`, optional): Minimum number of shared students a pair of classes must have to compute a correlation. Defaults to 20.
        directedCorr (:obj:`bool`, optional): Whether or not to include data specific to students who took class A before B, vice versa, and concurrently. Defaults to 'False'.
        detailed (:obj:`bool`, optional): Whether or not to include means of student grades, normalized grades, and standard deviations used. Defaults to 'False'.

    """
    if not filename.endswith('.csv'):
      filename = "".join((filename, '.csv'))
    self.getCorrelationsWithMinNSharedStudents(nSharedStudents=nStudents, directed=directedCorr, classDetails = detailed).to_csv(filename, index=False)

  def exportCorrelationsWithAvailableClasses(self, filename = 'CorrelationOutput_EDMLIB.csv'):
    result = self.getCorrelationsWithMinNSharedStudents()
    if result:
      result.to_csv(filename, index=False)

  def getCorrelationsOfAvailableClasses(self):
    return classCorrelationData(self.getCorrelationsWithMinNSharedStudents())

  def getListOfClassCodes(self):
    """Returns a list of unique class codes currently in the dataset from the 'classCode' column, which is the conjoined 'classDept' and 'classNumber' columns by default (e.g. 'Psych1000' from 'Psych' and '1000').

    Returns:
        :obj:`list`: List of unique class codes in the dataset.

    """
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      return
    return self.getUniqueValuesInColumn(self.CLASS_CODE_COLUMN)

  def getNormalizationColumn(self):
    """Used internally. Creates a normalization column 'norm' that is a "normalization" of grades recieved in specific classes. 
    This is equivelant to the grade given to a student minus the mean grade in that class, all divided by the standard 
    deviation of grades in that class.
    
    """
    print('Getting normalization column...')
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
    if self.GPA_STDDEV_COLUMN not in self.df.columns:
      self.getGPADeviations()
    if self.GPA_MEAN_COLUMN not in self.df.columns:
      self.getGPAMeans()
    if not all(item in self.df.columns for item in [self.CLASS_CODE_COLUMN, self.GPA_STDDEV_COLUMN, self.GPA_MEAN_COLUMN]):
      return
    begin = self.getEntryCount()
    self.filterByGpaDeviationMoreThan(0.001)
    self.df[self.NORMALIZATION_COLUMN] = (self.df[self.FINAL_GRADE_COLUMN].values - self.df[self.GPA_MEAN_COLUMN].values) / self.df[self.GPA_STDDEV_COLUMN].values

  def filterColumnToValues(self, col, values = []):
    """Filters dataset to only include rows that contain any of the given values in the given column.

    Args:
        col (:obj:`str`): Name of the column to filter.
        values (:obj:`list`): Values to filter to.
        
    """
    if not self.__requiredColumnPresent(col):
        return
    self.printEntryCount()
    if all([isinstance(x,str) for x in values]):
      lowered = [x.lower() for x in values]
      possibilities = "|".join(lowered)
      loweredCol = self.df[col].str.lower()
      self.df = self.df.loc[loweredCol.str.contains(possibilities)]
    else:
      self.df = self.df.loc[np.in1d(self.df[col],values)]
    self.df.reset_index(inplace=True, drop=True)
    self.printEntryCount()

  def filterToMultipleMajorsOrClasses(self, majors = [], classes = []):
    """Reduces the dataset to only include entries of certain classes and/or classes in certain majors. This function is 
    inclusive; if a class in 'classes' is not of a major defined in 'majors', the class will still be included, and 
    vice-versa.

    Note:
        The 'classDept' column as set by :obj:`defineWorkingColumns` must be defined in your dataset to filter by major.

    Args:
        majors (:obj:`list`, optional): List of majors to include. Filters by the 'classDept' column.
        classes (:obj:`list`, optional): List of classes to include. Filters by the 'classCode' column, or the conjoined version of 'classDept' and 'classNumber' columns.

    """
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      return
    self.printEntryCount()
    self.df = self.df.loc[(np.in1d(self.df[self.CLASS_CODE_COLUMN], classes)) | (np.in1d(self.df[self.CLASS_DEPT_COLUMN], majors))]
    self.df.reset_index(inplace=True, drop=True)
    self.printEntryCount()
  
  def filterStudentsByMajors(self, majors):
    """Filters the dataset down to students who were ever recorded as majoring in one of the given majors.

    Args:
      majors (:obj:`list`, optional): List of student majors to include when finding matching students. Filters by the 'studentMajor' column.

    """
    students = self.getUniqueValuesInColumn(self.STUDENT_ID_COLUMN)
    matchingRecords = self.df.loc[self.df[self.STUDENT_MAJOR_COLUMN].isin(majors)]
    validStudents = matchingRecords[self.STUDENT_ID_COLUMN].unique()
    self.df = self.df.loc[self.df[self.STUDENT_ID_COLUMN].isin(validStudents)]
    self.df.reset_index(inplace=True, drop=True)

  def getGPADeviations(self):
    """Used internally. Makes a new column called 'gpaStdDeviation', the standard deviation of grades of the respective 
    class for each entry.
    
    """
    print('Getting grade deviations by class...')
    if not self.__requiredColumnPresent(self.FINAL_GRADE_COLUMN):
        return
    self.dropMissingValuesInColumn(self.FINAL_GRADE_COLUMN)
    self.convertColumnToNumeric(self.FINAL_GRADE_COLUMN)
    if self.CLASS_ID_AND_TERM_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
    if not self.__requiredColumnPresent(self.CLASS_ID_AND_TERM_COLUMN):
        return
    temp = self.df.loc[:, [self.CLASS_ID_AND_TERM_COLUMN, self.FINAL_GRADE_COLUMN]]
    standardDev = temp.groupby(self.CLASS_ID_AND_TERM_COLUMN).std()
    standardDev.rename(columns={self.FINAL_GRADE_COLUMN:self.GPA_STDDEV_COLUMN}, inplace=True)
    self.df = pd.merge(self.df,standardDev, on=self.CLASS_ID_AND_TERM_COLUMN)

  def getGPAMeans(self):
    """Used internally. Makes a new column called 'gpaMean', the mean of grades recieved in the respective class of each entry.
    
    """
    if not self.__requiredColumnPresent(self.FINAL_GRADE_COLUMN):
        return
    self.dropMissingValuesInColumn(self.FINAL_GRADE_COLUMN)
    self.convertColumnToNumeric(self.FINAL_GRADE_COLUMN)
    if self.CLASS_ID_AND_TERM_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
    if not self.__requiredColumnPresent(self.CLASS_ID_AND_TERM_COLUMN):
        return
    temp = self.df.loc[:, [self.CLASS_ID_AND_TERM_COLUMN, self.FINAL_GRADE_COLUMN]]
    meanGpas = temp.groupby(self.CLASS_ID_AND_TERM_COLUMN).mean()
    meanGpas.rename(columns={self.FINAL_GRADE_COLUMN:self.GPA_MEAN_COLUMN}, inplace=True)
    self.df = pd.merge(self.df,meanGpas, on=self.CLASS_ID_AND_TERM_COLUMN)

  def filterByGpaDeviationMoreThan(self, minimum, outputDropped = False, droppedCSVName = 'droppedData.csv'):
    """Filters data to only include classes which have a standard deviation more than or equal to a given minimum (0.0 to 4.0 scale).

    Args:
        minimum (:obj:`float`): Minimum standard deviation of grades a class must have.
        outputDropped (:obj:`bool`, optional): Whether to output the dropped data to a file. Default is False.
        droppedCSVName (:obj:`str`, optional): Name of file to output dropped data to. Default is 'droppedData.csv`.

    """
    if self.GPA_STDDEV_COLUMN not in self.df.columns:
      self.getGPADeviations()
    if not self.__requiredColumnPresent(self.GPA_STDDEV_COLUMN):
        return
    if outputDropped:
      self.df[self.df[self.GPA_STDDEV_COLUMN] < minimum].to_csv(droppedCSVName, index=False)
    start = self.getEntryCount() 
    self.df = self.df[self.df[self.GPA_STDDEV_COLUMN] >= minimum]
    if start > self.getEntryCount():
      print('filtering by grade standard deviation of '+ str(minimum) +', ' + str(start - self.getEntryCount()) + ' entries dropped')

  def filterToSpecificValueInColumn(self, column, value):
    if not self.__requiredColumnPresent(column):
        return
    self.df = self.df[self.df[column] == value]
  
  def removeDuplicatesInColumn(self, column):
    if not self.__requiredColumnPresent(column):
        return
    self.df.drop_duplicates(subset=column, inplace=True)

  def countDuplicatesInColumn(self, column, term):
    if not self.__requiredColumnPresent(column):
        return
    return self.df[column].value_counts()[term]

  def countUniqueValuesInColumn(self, column):
    if not self.__requiredColumnPresent(column):
        return
    return self.df[column].nunique()

  def getUniqueValuesInColumn(self, column):
    if not self.__requiredColumnPresent(column):
        return
    return self.df[column].unique()

  def printUniqueValuesInColumn(self, column):
    """Prints to console the unique variables in a given column.

    Args:
        column (:obj:`str`): Column to get unique variables from.
    
    """
    if not self.__requiredColumnPresent(column):
        return
    print(self.df[column].unique())

  def getEntryCount(self):
    return len(self.df.index)

  def printEntryCount(self):
    """Prints to console the number of entries (rows) in the current dataset.
    
    """
    print(str(len(self.df.index)) + ' entries')

  def printFirstXRows(self, rows):
    """Prints to console the first X number of rows from the dataset.
    
    Args:
        rows (:obj:`int`): Number of rows to print from the dataset.

    """
    print(self.df.iloc[:rows])
  
  def removeRowsWithNullInColumn(self, column):
    if not self.__requiredColumnPresent(column):
        return
    self.df = self.df[self.df[column] != ' ']

  def removeRowsWithZeroInColumn(self, column):
    if not self.__requiredColumnPresent(column):
        return
    self.df = self.df[self.df[column] != 0]

  def getPandasDataFrame(self):
    """Returns the pandas dataframe of the dataset.

    Returns:
        :obj:`pandas.dataframe`: Dataframe of the current dataset.

    """
    return self.df

  def exportCSV(self, fileName = 'csvExport.csv'):
    """Export the current state of the dataset to a :obj:`.CSV` file.
    
    Args:
        fileName (:obj:`str`, optional): Name of the file to export. Defaults to 'csvExport.csv'.

    """
    self.df.to_csv(fileName, index=False)

  def __requiredColumnPresent(self, column):
    if column not in self.df.columns:
      if edmApplication:
        print("Error: required column '" + column + "' not present in dataset. Fix by right clicking / setting columns.")
      else:
        print("Error: required column '" + column + "' not present in dataset. Fix or rename with the 'defineWorkingColumns' function.")
      return False
    return True

class classCorrelationData:
  """Class for manipulating and visualizing pearson correlations generated by the gradeData class.

    Attributes:
        df (:obj:`pandas.dataframe`): dataframe containing all correlational data.
        sourceFile (:obj:`str`): Name of source .CSV file with correlational data.

  """
  df = None
  sourceFile = ""

  def __init__(self,sourceFileOrDataFrame):
    """Class constructor, creates an instance of the class given a .CSV file or pandas dataframe. Typically should only be used manually with correlation files outputted by the gradeData class.

    Used with classCorrelationData('fileName.csv') or classCorrelationData(dataFrameVariable).

    Args:
        sourceFileOrDataFrame (:obj:`object`): name of the .CSV file (extension included) in the same path or pandas dataframe variable. Dataframes are copied so as to not affect the original variable.

    """
    if type(sourceFileOrDataFrame).__name__ == 'str':
      self.sourceFile = sourceFileOrDataFrame
      self.df = pd.read_csv(self.sourceFile)

    elif type(sourceFileOrDataFrame).__name__ == 'DataFrame':
      if edmApplication:
        self.df = sourceFileOrDataFrame
      else:
        self.df = sourceFileOrDataFrame.copy()
  
  def getClassesUsed(self):
    return self.df['course1'].unique()

  def getNumberOfClassesUsed(self):
    return self.df['course1'].nunique()

  def printUniqueValuesInColumn(self, column):
    print(self.df[column].unique())

  def printClassesUsed(self):
    """Prints to console the classes included in the correlations.
    
    """
    self.printUniqueValuesInColumn('course1')

  def getEntryCount(self):
    return len(self.df.index)

  def printEntryCount(self):
    """Prints to console the number of entries (rows) in the current dataset.
    
    """
    print(str(len(self.df.index)) + ' entries')

  def printFirstXRows(self, rows):
    """Prints to console the first X number of rows from the dataset.
    
    Args:
        rows (:obj:`int`): Number of rows to print from the dataset.

    """
    print(self.df.iloc[:rows])
  
  def printMajors(self):
    """Prints to console the majors of the classes present in the correlational data.
    
    Note:
        The 'classDept' column as set by :obj:`defineWorkingColumns` must have been defined in your dataset to print majors.

    """
    courses = self.getClassesUsed().tolist()
    # below is a regex expression to only include non-numeric characters, inside a set comprehension
    majors = {re.findall('\A\D+', course)[0] for course in courses}
    print(majors)

  def filterColumnToValues(self, col, values = []):
    """Filters dataset to only include rows that contain any of the given values in the given column.

    Args:
        col (:obj:`str`): Name of the column to filter.
        values (:obj:`list`): Values to filter to.
        
    """
    self.printEntryCount()
    self.df = self.df.loc[np.in1d(self.df[col],values)]
    self.df.reset_index(inplace=True, drop=True)
    self.printEntryCount()

  def exportCSV(self, fileName = 'csvExport.csv'):
    """Export the current state of the dataset to a :obj:`.CSV` file.
    
    Args:
        fileName (:obj:`str`, optional): Name of the file to export. Defaults to 'csvExport.csv'.

    """
    self.df.to_csv(fileName, index=False)

  def filterToMultipleMajorsOrClasses(self, majors = [], classes = [], twoWay = True):
    """Reduces the dataset to only include entries of certain classes and/or classes in certain majors. This function is 
    inclusive; if a class in 'classes' is not of a major defined in 'majors', the class will still be included, and 
    vice-versa.

    Note:
        The 'classDept' column as set by :obj:`defineWorkingColumns` must have been defined in your dataset to filter by major.

    Args:
        majors (:obj:`list`, optional): List of majors to include. Filters by the 'classDept' column in the original dataset.
        classes (:obj:`list`, optional): List of classes to include. Filters by the 'classCode' column in the original dataset, or the conjoined version of 'classDept' and 'classNumber' columns.
        twoWay (:obj:`bool`, optional): Whether both classes in the correlation must be in the given majors / classes, or only one of them. Set to :obj:`True`, or both classes, by default.

    """
    if twoWay:
      self.df = self.df.loc[((self.df['course1'].isin(classes)) | (self.df['course1'].apply(lambda course: re.findall('\A\D+', course)[0])).isin(majors)) & ((self.df['course2'].isin(classes)) | (self.df['course2'].apply(lambda course: re.findall('\A\D+', course)[0]).isin(majors)))]
    else:
      self.df = self.df.loc[((self.df['course1'].isin(classes)) | (self.df['course1'].apply(lambda course: re.findall('\A\D+', course)[0])).isin(majors)) | ((self.df['course2'].isin(classes)) | (self.df['course2'].apply(lambda course: re.findall('\A\D+', course)[0]).isin(majors)))]
    self.df.reset_index(inplace=True, drop=True)

  def substituteSubStrInColumn(self, column, subString, substitute):
    """Replace a substring in a given column.

      Args:
        column (:obj:`str`): Column to replace substring in.
        subString (:obj:`str`): Substring to replace.
        substitute (:obj:`str`): Replacement of the substring.

    """
    self.convertColumnToString(column)
    self.df[column] = self.df[column].str.replace(subString, substitute)

  def chordGraphByMajor(self, coefficient = 0.5, pval = 0.05, outputName = 'majorGraph', outputSize = 200, imageSize = 300, showGraph = True, outputImage = True):
    """Creates a chord graph between available majors through averaging and filtering both correlation coefficients and P-values. Outputs to an html file, PNG file, and saves the underlying data by default.

    Note:
        The 'classDept' column as set by :obj:`defineWorkingColumns` must have been defined in your dataset to filter by major.

    Args:
        coefficient (:obj:`float`, optional): Minimum correlation coefficient to filter correlations by.
        pval (:obj:`float`, optional): Maximum P-value to filter correlations by.
        outputName (:obj:`str`, optional): First part of the outputted file names, e.g. fileName.csv, fileName.html, etc.
        outputSize (:obj:`int`, optional): Size (units unknown) of html graph to output. 200 by default.
        imageSize (:obj:`int`, optional): Size (units unknown) of image of the graph to output. 300 by default. Increase this if node labels are cut off.
        showGraph (:obj:`bool`, optional): Whether or not to open a browser and display the interactive graph that was created. Defaults to :obj:`True`.
        outputImage (:obj:`bool`, optional): Whether or not to export an image of the graph. Defaults to :obj:`True`. 
    
    """
    hv.output(size=outputSize)
    majorFiltered = self.df.copy()
    majorFiltered['course1'] = majorFiltered['course1'].apply(lambda course: re.findall('\A\D+', course)[0])
    majorFiltered['course2'] = majorFiltered['course2'].apply(lambda course: re.findall('\A\D+', course)[0])
    majors = majorFiltered['course1'].unique().tolist()
    majors.sort()
    majorCorrelations = []
    usedMajors = []
    majorFiltered['corr'] = pd.to_numeric(majorFiltered['corr'])
    majorFiltered['P-value'] = pd.to_numeric(majorFiltered['P-value'])
    majorFiltered['#students'] = pd.to_numeric(majorFiltered['#students'])
    count = 0
    for major in majors:
      count += 1
      print(str(count) + ' / ' + str(len(majors)) + ' majors')
      filteredToMajor = majorFiltered.loc[majorFiltered['course1'] == major]
      connectedMajors = filteredToMajor['course2'].unique().tolist()
      for targetMajor in connectedMajors:
        filteredToMajorPair = filteredToMajor.loc[filteredToMajor['course2'] == targetMajor]
        avgCorr = int(filteredToMajorPair['corr'].mean() * 100)
        avgPVal = filteredToMajorPair['P-value'].mean()
        avgStudents = filteredToMajorPair['#students'].mean()
        if avgCorr > (coefficient * 100) and major != targetMajor and avgPVal < pval:
          if (targetMajor, major) not in usedMajors:
            usedMajors.append((major, targetMajor))
            majorCorrelations.append((major, targetMajor, avgCorr, avgPVal, avgStudents))

    if len(majorCorrelations) == 0:
      print('Error: no valid correlations found.')
      return
    print(str(len(majorCorrelations)) + ' valid major correlations found.')
    output = pd.DataFrame(majorCorrelations, columns=('source', 'target', 'corr', 'P-value', '#students'))
    newMajors = set(output['source'])
    newMajors.update(output['target'])
    sortedMajors = sorted(list(newMajors))
    nodes = pd.DataFrame(sortedMajors, columns = ['name'])
    output['source'] = output['source'].apply(lambda major: nodes.index[nodes['name'] == major][0])
    output['target'] = output['target'].apply(lambda major: nodes.index[nodes['name'] == major][0])

    output.to_csv(outputName + '.csv', index=False)
    hvNodes = hv.Dataset(nodes, 'index')
    chord = hv.Chord((output, hvNodes)).select(value=(5, None))
    chord.opts(
        opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('source').str(), 
                  labels='name', node_color=dim('index').str()))
    graph = hv.render(chord)
    output_file(outputName + '.html', mode='inline')
    save(graph)
    if showGraph:
      show(graph)
    chord.opts(toolbar=None)
    if outputImage:
      hv.output(size=imageSize)
      export_png(hv.render(chord), filename=outputName + '.png')
    hv.output(size=outputSize)

  def getNxGraph(self, minCorr = None):
    """Returns a NetworkX graph of the correlational data, where the nodes are classes and the weights are the correlations.

      Args:
        minCorr (:obj:`float`, optional): Minimum correlation between classes for an edge to be included on the graph. Should be in the 0.0-1.0 range. Defaults to :obj:`None` (or do not filter).

    """
    if minCorr is None:
      print('minCorr none')
      return nx.from_pandas_edgelist(self.df, 'course1', 'course2', 'corr')
    self.df['corr'] = pd.to_numeric(self.df['corr'])
    filtered = self.df.loc[self.df['corr'] >= minCorr]
    return nx.from_pandas_edgelist(filtered, 'course1', 'course2', 'corr')

  def getCliques(self, minCorr = None, minSize = 2):
    """Returns a list of lists / cliques present in the correlational data. Cliques are connected sub-graphs in the larger overall graph.

    Args:
        minCorr (:obj:`None` or :obj:`float`, optional): Minimum correlation to consider a correlation an edge on the graph. 'None', or ignored, by default.
        minSize (:obj:`int`, optional): Minimum number of nodes to look for in a clique. Default is 2.

    """
    graph = self.getNxGraph(minCorr)
    cliques = list(nx.find_cliques(graph))
    return sorted(filter(lambda clique: len(clique) >= minSize, cliques))

  def outputCliqueDistribution(self, minCorr = None, countDuplicates = False, makeHistogram = False, fileName = 'cliqueHistogram', graphTitle = 'Class Correlation Cliques', logScale = False):
    """Outputs the clique distribution from the given correlation data. Prints to console by default, but can also optionally export a histogram.

    Args:
        minCorr (:obj:`None` or :obj:`float`, optional): Minimum correlation to consider a correlation an edge on the graph. 'None', or ignored, by default.
        countDuplicates (:obj:`bool`, optional): Whether or not to count smaller sub-cliques of larger cliques as cliques themselves. :obj:`False` by default.
        makeHistogram (:obj:`bool`, optional): Whether or not to generate a histogram. False by default.
        fileName (:obj:`str`, optional): File name to give exported histogram files. 'cliqueHistogram' by default.
        graphTitle (:obj:`str`, optional): Title displayed on the histogram. 'Class Correlation Cliques' by default.
        logScale (:obj:`bool`, optional): Whether or not to output graph in Log 10 scale on the y-axis. Defaults to :obj:`False`.

    """
    cliques = self.getCliques(minCorr = minCorr)
    largestClique = len(max(cliques, key = len))
    weight = []
    for k in range(2, largestClique+1):
      count = 0
      for clique in cliques:
        if len(clique) == k:
            count += 1
        elif countDuplicates and len(clique) > k:
            count += len(list(itertools.combinations(clique, k)))
      weight.append(count)
      print('Size ' + str(k) + ' cliques: ' + str(count))
    if makeHistogram:
      # cliqueCount = [len(clique) for clique in cliques]
      # frequencies, edges = np.histogram(cliqueCount, largestClique - 1, (2, largestClique))
      cliqueCount = range(2, largestClique+1)
      frequencies, edges = np.histogram(a=cliqueCount, bins=largestClique - 1, range=(2, largestClique), weights=weight)
      #print('Values: %s, Edges: %s' % (frequencies.shape[0], edges.shape[0]))
      ylbl = 'Number of Cliques'
      if logScale:
        frequencies = [math.log10(freq) for freq in frequencies]
        ylbl += ' (log 10 scale)'
      histo = hv.Histogram((edges, frequencies))
      histo.opts(opts.Histogram(xlabel='Number of Classes in Clique', ylabel=ylbl, title=graphTitle))
      hv.output(size=125)
      subtitle = 'n=' + str(self.getEntryCount())
      if minCorr:
        subtitle = 'corr >= ' + str(minCorr) + ', ' + subtitle

      graph = hv.render(histo)
      graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="10pt"), 'above')
      output_file(fileName + '.html', mode='inline')
      save(graph)
      show(graph)
      if not edmApplication:
        histo.opts(toolbar=None)
        graph = hv.render(histo)
        graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="10pt"), 'above')
        export_png(graph, filename=fileName + '.png')

  def makeMissingValuesNanInColumn(self, column):
    if not self.__requiredColumnPresent(column):
      return
    self.df[column].replace(' ', np.nan, inplace=True)

  def removeNanInColumn(self, column):
    if not self.__requiredColumnPresent(column):
      return
    self.df.dropna(subset=[column], inplace=True)
    self.df.reset_index(inplace = True, drop=True)

  def dropMissingValuesInColumn(self, column):
    """Removes rows in the dataset which have missing data in the given column.

      Args:
        column (:obj:`str`): Column to check for missing values in.

    """
    if not self.__requiredColumnPresent(column):
      return
    self.makeMissingValuesNanInColumn(column)
    self.removeNanInColumn(column)      

  def __requiredColumnPresent(self, column):
    if column not in self.df.columns:
      if edmApplication:
        print("Error: required column '" + column + "' not present in dataset. Fix by right clicking / setting columns.")
      else:
        print("Error: required column '" + column + "' not present in dataset. Fix or rename with the 'defineWorkingColumns' function.")
      return False
    return True

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        # self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.message = QtWidgets.QLabel()
        self.statusBar().addWidget(self.message)
        self.settings = QSettings('EDM Lab', 'EDM Program')
        self.setWindowTitle("EDM Program")
        # self.settings.setValue("lastFile", None)
        if self.settings.value("lastFile", None):
          if os.path.isfile(self.settings.value("lastFile", None)):
            self.setWindowTitle("EDM - " + self.settings.value("lastFile",None))
        self.terminal = sys.stdout
        sys.stdout = self
        self.resize(self.settings.value("size", QSize(640, 480)))
        self.move(self.settings.value("pos", QPoint(50, 50)))
        self.correlationFile = False
        self.grades = False
        
        self.first = False
        self.reload()
        self.first = True
        self.setCentralWidget(self.tableView)
        self.menubar = self.menuBar()
        self.setGeneralButtons()
        
        # menubar = QMenuBar()
        
        # exitAct.setShortcut('Ctrl+Q')
        # exitAct.setStatusTip('Exit application')
        # exitAct.triggered.connect(self.close)


        # self.pushButtonLoad = QtWidgets.QPushButton(self)
        # self.pushButtonLoad.setText("Load Csv File")
        # self.pushButtonLoad.clicked.connect(self.on_pushButtonLoad_clicked)

        # self.pushButtonWrite = QtWidgets.QPushButton(self)
        # self.pushButtonWrite.setText("Save Csv File")
        # self.pushButtonWrite.clicked.connect(self.on_pushButtonWrite_clicked)

        # self.layoutVertical = QtWidgets.QVBoxLayout(self)
        # self.layoutVertical.addWidget(self.tableView)
        # self.layoutVertical.addWidget(self.pushButtonLoad)
        # self.layoutVertical.addWidget(self.pushButtonWrite)

        if self.settings.value("lastFile", None):
          if os.path.isfile(self.settings.value("lastFile", None)):
            self.loadCsv(self.settings.value("lastFile", None))
            print('File opened.')
            self.doColumnThings()
            self.setGeneralButtons()

        self.threadpool = QThreadPool()
        self.show()

    def eventFilter(self, source, event):
        if (event.type() == QtCore.QEvent.KeyPress and
            event.matches(QtGui.QKeySequence.Copy)):
            self.copySelection()
            return True
        return super(MyWindow, self).eventFilter(source, event)

    def copySelection(self):
        selection = self.tableView.selectedIndexes()
        if selection:
            rows = sorted(index.row() for index in selection)
            columns = sorted(index.column() for index in selection)
            rowcount = rows[-1] - rows[0] + 1
            colcount = columns[-1] - columns[0] + 1
            table = [[''] * colcount for _ in range(rowcount)]
            for index in selection:
                row = index.row() - rows[0]
                column = index.column() - columns[0]
                table[row][column] = index.data()
            stream = io.StringIO()
            csv.writer(stream, delimiter='\t').writerows(table)
            QtWidgets.qApp.clipboard().setText(stream.getvalue())  

    def reload(self):
        self.model = QtGui.QStandardItemModel(self)
        # for key in self.settings.allKeys():
        #   print(str(key) + ' ' + str(self.settings.value(key, None)))

        self.tableView = QtWidgets.QTableView(self)
        self.tableView.installEventFilter(self)
        self.tableView.setModel(self.model)
        self.tableView.setSortingEnabled(True)
        self.tableView.horizontalHeader().setStretchLastSection(True)
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tableView.horizontalHeader().setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tableView.horizontalHeader().customContextMenuRequested.connect(self.onColumnRightClick)
        if isinstance(self, MyWindow):
          self.setCentralWidget(self.tableView)
        elif isinstance(self, csvPreview):
          self.layout.itemAt(0).widget().setParent(None)
          self.layout.insertWidget(0,self.tableView)
          # self.layout.addWidget(self.tableView)
          # self.layout.addWidget(self.buttonBox)
        if self.first:
            try:
                self.df = self.grades.df
                self.model = TableModel(self.grades.df)
                self.tableView.setModel(self.model)
                self.getLastKnownColumns()
            except: 
                pass
    def setGeneralButtons(self):
        self.menubar.clear()
        self.menubar = self.menuBar()
        self.fileMenu = self.menubar.addMenu('File')
        openFile = QAction('Open', self)
        openFile.triggered.connect(self.on_pushButtonLoad_clicked)
        saveAs = QAction('Save As', self)
        saveAs.triggered.connect(self.on_pushButtonWrite_clicked)
        self.fileMenu.addAction(openFile)
        self.fileMenu.addAction(saveAs)

        self.menubar.setNativeMenuBar(False)
        self.setMenuBar(self.menubar)

        if self.settings.value("lastFile", None) and not self.correlationFile:
          if os.path.isfile(self.settings.value("lastFile", None)):
            self.correlationMenu = self.menubar.addMenu('Correlations')
            exportCorr = QAction('Export Class Correlations', self)
            exportCorr.triggered.connect(self.exportCorrelations)
            self.correlationMenu.addAction(exportCorr)
            exportGPAS = QAction('Export GPA Distribution', self)
            exportGPAS.triggered.connect(self.exportGPADistribution)
            self.correlationMenu.addAction(exportGPAS)
            sankeyTrackNew = QAction('Export Course Track Graph', self)
            sankeyTrackNew.triggered.connect(self.makeSankeyTrackNew)
            self.correlationMenu.addAction(sankeyTrackNew)
            sankeyTrack = QAction('Export Course Track Graph (alternate)', self)
            sankeyTrack.triggered.connect(self.makeSankeyTrack)
            self.correlationMenu.addAction(sankeyTrack)
            sankeyTrack2 = QAction('Export Track Graph (Experimental)', self)
            sankeyTrack2.triggered.connect(self.makeSankeyTrackAdvanced)
            self.correlationMenu.addAction(sankeyTrack2)
            self.filterMenu = self.menubar.addMenu('Filters')
            byClassOrMajor = QAction('Filter by Classes / Class Depts', self)
            byClassOrMajor.triggered.connect(self.filterByClassOrMajor)
            self.filterMenu.addAction(byClassOrMajor)
            byStdntMajor = QAction('Filter by Student Majors', self)
            byStdntMajor.triggered.connect(self.filterByStudentMajor)
            self.filterMenu.addAction(byStdntMajor)
            byGPADev = QAction('Filter Classes by Grade Deviation', self)
            byGPADev.triggered.connect(self.filterGPADeviation)
            self.filterMenu.addAction(byGPADev)
            self.calcMenu = self.menubar.addMenu('Calculations')
            gpaMean = QAction('Calculate Class Grade Means', self)
            gpaMean.triggered.connect(self.gpaMeanCol)
            self.calcMenu.addAction(gpaMean)
            gpaDev = QAction('Calculate Class Grade Std Deviations', self)
            gpaDev.triggered.connect(self.gpaDevCol)
            self.calcMenu.addAction(gpaDev)
            norms = QAction('Calculate Grade Normalizations', self)
            norms.triggered.connect(self.normCol)
            self.calcMenu.addAction(norms)
            instructorEffect = QAction('Calculate Instructor Effectiveness', self)
            instructorEffect.triggered.connect(self.instructorEffectiveness)
            self.calcMenu.addAction(instructorEffect)
            instructorEffectAll = QAction('Calculate Instructor Effectiveness (All)', self)
            instructorEffectAll.triggered.connect(self.instructorEffectivenessAll)
            self.calcMenu.addAction(instructorEffectAll)
            setColumns = QAction('Set Columns', self)
            setColumns.triggered.connect(self.designateColumns)
            self.menubar.addAction(setColumns)
            predict = QAction('Grade Predict', self)
            predict.triggered.connect(self.gradePredict)
            self.menubar.addAction(predict)

        elif self.settings.value("lastFile", None):
          if os.path.isfile(self.settings.value("lastFile", None)):
            self.correlationMenu = self.menubar.addMenu('Correlations')
            majorChord = QAction('Export Chord Graph by Major', self)
            majorChord.triggered.connect(self.exportMajorChord)
            self.correlationMenu.addAction(majorChord)
            cliqueHisto = QAction('Export Clique Histogram', self)
            cliqueHisto.triggered.connect(self.exportCliqueHisto)
            self.correlationMenu.addAction(cliqueHisto)
            self.filterMenu = self.menubar.addMenu('Filters')
            byClassOrMajor = QAction('Filter to Classes / Class Depts', self)
            byClassOrMajor.triggered.connect(self.filterByClassOrMajorCorr)
            self.filterMenu.addAction(byClassOrMajor)
        if self.settings.value("lastFile", None):
          if os.path.isfile(self.settings.value("lastFile", None)):
            stats = QAction('Show Stats', self)
            stats.triggered.connect(self.getStats)
            self.menubar.addAction(stats)
            getOriginal = QAction('Reload Original File', self)
            getOriginal.triggered.connect(self.originalReload)
            self.fileMenu.addAction(getOriginal)

    def doColumnThings(self):
        self.columnSelected = None
        self.rcMenu=QMenu(self)
        if not self.correlationFile:
            self.setMenu = self.rcMenu.addMenu('Set Column')
            setDept = QAction('Set Class Department Column', self)
            setDept.triggered.connect(self.designateDept)
            self.setMenu.addAction(setDept)
            setClassNumber = QAction('Set Class Number Column', self)
            setClassNumber.triggered.connect(self.designateClssNmbr)
            self.setMenu.addAction(setClassNumber)
            setCID = QAction('Set Class ID Column', self)
            setCID.triggered.connect(self.designateCID)
            self.setMenu.addAction(setCID)
            setSID = QAction('Set Student ID Column', self)
            setSID.triggered.connect(self.designateSID)
            self.setMenu.addAction(setSID)
            setGrades = QAction('Set Numeric Grade Column', self)
            setGrades.triggered.connect(self.designateGrades)
            self.setMenu.addAction(setGrades)
            setTerm = QAction('Set Term Column', self)
            setTerm.triggered.connect(self.designateTerms)
            self.setMenu.addAction(setTerm)
            setStdntMjr = QAction('Set Student Major Column', self)
            setStdntMjr.triggered.connect(self.designateStdntMjr)
            self.setMenu.addAction(setStdntMjr)
            setCredits = QAction('Set Class Credits Column (optional)', self)
            setCredits.triggered.connect(self.designateCredits)
            self.setMenu.addAction(setCredits)
            setFID = QAction('Set Faculty ID Column (optional)', self)
            setFID.triggered.connect(self.designateFID)
            self.setMenu.addAction(setFID)
            setClassCode = QAction('Set Class Code Column (optional)', self)
            setClassCode.triggered.connect(self.designateClassCode)
            self.setMenu.addAction(setClassCode)
            self.getLastKnownColumns()
            renColumn = QAction('Rename Column...', self)
            renColumn.triggered.connect(self.renameColumn)
            self.rcMenu.addAction(renColumn)
        self.substituteMenu = self.rcMenu.addMenu('Substitute')
        substituteInColumn = QAction('Substitute in Column...', self)
        substituteInColumn.triggered.connect(self.strReplace)
        self.substituteMenu.addAction(substituteInColumn)
        dictReplaceInColumn = QAction('Substitute Many Values...', self)
        dictReplaceInColumn.triggered.connect(self.dictReplace)
        self.substituteMenu.addAction(dictReplaceInColumn)
        dictReplaceTxt = QAction('Use substitution file...', self)
        dictReplaceTxt.triggered.connect(self.dictReplaceFile)
        self.substituteMenu.addAction(dictReplaceTxt)
        self.fNumMenu = self.rcMenu.addMenu('Filter / Numeric Operations')
        valFilter = QAction('Filter Column to Value(s)...', self)
        valFilter.triggered.connect(self.filterColByVals)
        self.fNumMenu.addAction(valFilter)
        NumericFilter = QAction('Filter Column Numerically...', self)
        NumericFilter.triggered.connect(self.filterColNumeric)
        self.fNumMenu.addAction(NumericFilter)
        absFilter = QAction('Make Absolute Values', self)
        absFilter.triggered.connect(self.absCol)
        self.fNumMenu.addAction(absFilter)
        avStats = QAction('Get Mean / Med. / Mode', self)
        avStats.triggered.connect(self.avCol)
        self.fNumMenu.addAction(avStats)
        roundFilter = QAction('Round Column...', self)
        roundFilter.triggered.connect(self.roundCol)
        self.fNumMenu.addAction(roundFilter)
        NAFilter = QAction('Drop Undefined Values in Column', self)
        NAFilter.triggered.connect(self.removeNaInColumn)
        self.rcMenu.addAction(NAFilter)
        # if not self.correlationFile:
        deleteColumn = QAction('Delete Column (permanent)', self)
        deleteColumn.triggered.connect(self.delColumn)
        self.rcMenu.addAction(deleteColumn)

    def onColumnRightClick(self, QPos=None):       
        parent=self.sender()
        pPos=parent.mapToGlobal(QtCore.QPoint(0, 0))
        mPos=pPos+QPos
        column = self.tableView.horizontalHeader().logicalIndexAt(QPos)
        label = self.model.headerData(column, Qt.Horizontal, Qt.DisplayRole)
        self.columnSelected = label
        self.rcMenu.move(mPos)
        self.rcMenu.show()

    def getLastKnownColumns(self):
      if self.settings.value("ClassID", None):
        self.designateCID(self.settings.value("ClassID", None))
      if self.settings.value("ClassCode", None):
        self.designateClassCode(self.settings.value("ClassCode", None))
      if self.settings.value("ClassNumber", None):
        self.designateClssNmbr(self.settings.value("ClassNumber", None))
      if self.settings.value("ClassDept", None):
        self.designateDept(self.settings.value("ClassDept", None))
      if self.settings.value("Grades", None):
        self.designateGrades(self.settings.value("Grades", None))
      if self.settings.value("StudentID", None):
        self.designateSID(self.settings.value("StudentID", None))
      if self.settings.value("StudentMajor", None):
        self.designateStdntMjr(self.settings.value("StudentMajor", None))
      if self.settings.value("Terms", None):
        self.designateTerms(self.settings.value("Terms", None))
      if self.settings.value("Credits", None):
        self.designateCredits(self.settings.value("Credits", None))
      if self.settings.value("FID", None):
        self.designateFID(self.settings.value("FID", None))

    def designateColumns(self):
        dlg = columnDialog(self, list(self.grades.df.columns.values))
        dlg.setWindowTitle("Set Columns")
        if dlg.exec():
          try:
              result = dlg.getInputs()
              self.designateDept(result[0])
              self.designateClssNmbr(result[1])
              self.designateCID(result[2])
              self.designateSID(result[3])
              self.designateGrades(result[4])
              self.designateTerms(result[5])
              self.designateStdntMjr(result[6])
              self.designateCredits(result[7])
              self.designateFID(result[8])
              self.designateClassCode(result[9])
          except Exception as e: 
              print(e)
              pass

    def gradePredict(self):
        if ((self.grades.CLASS_CODE_COLUMN in self.grades.df.columns) or ((self.grades.CLASS_DEPT_COLUMN in self.grades.df.columns) and (self.grades.CLASS_NUMBER_COLUMN in self.grades.df.columns) )) and (self.grades.FINAL_GRADE_COLUMN in self.grades.df.columns):
          dlg = gradePredictDialogue(self)
          dlg.setWindowTitle("Predict Grades")
          if dlg.exec():
            prior, future, mode = dlg.getInputs()
            self.grades.gradePredict(prior, future, mode)
        else:
          print('Error: Class dept/num or code and grade columns required. Fix with "set columns" in the top menu.')

    def getStats(self):
        dlg = statsDialogue(self)
        dlg.setWindowTitle("Dataframe Statistics")
        dlg.exec()
            # try:
            #   corr, pval, name = dlg.getInputs()
            #   worker = Worker(self.grades.chordGraphByMajor, corr, pval, name)
            #   self.threadpool.start(worker) 
            # except Exception as e: 
            #   print(e)
            #   pass

    def renameColumn(self):
        oldName = self.columnSelected
        dlg = renameColumnDialogue(self)
        dlg.setWindowTitle("Rename Column")
        if dlg.exec():
            try:
                newName = dlg.getInputs()
                self.grades.df.rename({oldName: newName}, axis=1, inplace=True)
                if self.settings.value("ClassID", None) == oldName:
                  self.designateCID(newName)
                if self.settings.value("ClassCode", None) == oldName:
                  self.designateClassCode(newName)
                if self.settings.value("ClassNumber", None) == oldName:
                  self.designateClssNmbr(newName)
                if self.settings.value("ClassDept", None) == oldName:
                  self.designateDept(newName)
                if self.settings.value("Grades", None) == oldName:
                  self.designateGrades(newName)
                if self.settings.value("StudentID", None) == oldName:
                  self.designateSID(newName)
                if self.settings.value("StudentMajor", None) == oldName:
                  self.designateStdntMjr(newName)
                if self.settings.value("Terms", None) == oldName:
                  self.designateTerms(newName)
                if self.settings.value("Credits", None) == oldName:
                  self.designateCredits(newName)
                if self.settings.value("FID", None) == oldName:
                  self.designateFID(newName)
            except Exception as e: 
                print(e)
                return

    def removeNaInColumn(self):
        try:
            firstNum = len(self.grades.df.index)
            self.grades.dropMissingValuesInColumn(self.columnSelected)
        except Exception as e: 
            print(e)
            return
        self.reload()
        print(str(firstNum - len(self.grades.df.index)) + ' undefined values dropped in ' + str(self.columnSelected) + ' column.')
    @QtCore.pyqtSlot()
    def filterColNumeric(self):
            try:
              self.removeNaInColumn()
              self.grades.df[self.columnSelected] = pd.to_numeric(self.grades.df[self.columnSelected], errors='raise')
            except:
              print('Error: Column is not numeric. Check values.')
              return
            dlg = filterNumsDialogue(self)
            dlg.setWindowTitle("Filter Column Numerically")
            if dlg.exec():
                try:
                    minVal, maxVal = dlg.getInputs()
                    # self.model.layoutAboutToBeChanged.emit()
                    filterOutput = 'Values in column '+str(self.columnSelected)
                    if not((minVal is None) or (maxVal is None)):
                      self.grades.df = self.grades.df.loc[(self.grades.df[self.columnSelected] >= minVal) & (self.grades.df[self.columnSelected] <= maxVal)]
                      filterOutput += ' filtered to min ' + str(minVal) + ' and max ' + str(maxVal)
                    elif not(minVal is None):
                      self.grades.df = self.grades.df.loc[self.grades.df[self.columnSelected] >= minVal]
                      filterOutput += ' filtered to min ' + str(minVal)
                    elif not(maxVal is None):
                      self.grades.df = self.grades.df.loc[self.grades.df[self.columnSelected] <= maxVal]
                      filterOutput += ' filtered to max ' + str(maxVal)
                    else:
                      filterOutput += ' not filtered.'
                    self.grades.df.reset_index(inplace = True, drop=True)
                    self.reload()
                    # self.model.layoutChanged.emit()
                    print(filterOutput)
                except Exception as e: 
                    print(e)
                    return
    @QtCore.pyqtSlot()
    def roundCol(self):
            try:
              self.model.layoutAboutToBeChanged.emit()
              self.removeNaInColumn()
              self.grades.df[self.columnSelected] = pd.to_numeric(self.grades.df[self.columnSelected], errors='raise')
              self.model.layoutChanged.emit()
            except:
              print('Error: Column is not numeric. Check values.')
              return
            dlg = roundDialogue(self)
            dlg.setWindowTitle("Round Column Values")
            if dlg.exec():
                try:
                    self.model.layoutAboutToBeChanged.emit()
                    decPlaces = dlg.getInputs()
                    self.grades.df[self.columnSelected] = self.grades.df[self.columnSelected].round(decPlaces)
                    # self.reload()
                    self.model.layoutChanged.emit()
                    print('Values in column '+str(self.columnSelected) +' rounded to ' + str(decPlaces) + ' decimal places.')
                except Exception as e: 
                    print(e)
                    return

    @QtCore.pyqtSlot()
    def avCol(self):
        try:
          self.model.layoutAboutToBeChanged.emit()
          self.removeNaInColumn()
          self.grades.df[self.columnSelected] = pd.to_numeric(self.grades.df[self.columnSelected], errors='raise')
          self.model.layoutChanged.emit()
          mean = self.grades.df[self.columnSelected].mean()
          med = self.grades.df[self.columnSelected].median()
          mode = self.grades.df[self.columnSelected].mode()
          stRnd = lambda x : str(round(x, 3))
          print(self.columnSelected + ' mean: ' + stRnd(mean) + '    median: ' + stRnd(med) + '    mode: ' + stRnd(mode.iloc[0]))
        except:
          print(self.columnSelected + ' mode: ' + str(self.grades.df[self.columnSelected].mode().iloc[0]))
          return

    @QtCore.pyqtSlot()
    def absCol(self):
        try:
          self.model.layoutAboutToBeChanged.emit()
          self.removeNaInColumn()
          self.grades.df[self.columnSelected] = pd.to_numeric(self.grades.df[self.columnSelected], errors='raise')
          self.grades.df[self.columnSelected] = self.grades.df[self.columnSelected].abs()
          self.model.layoutChanged.emit()
        except:
          print('Error: Column is not numeric. Check values.')
          return

    def filterColByVals(self):
        dlg = filterValsDialogue(self)
        dlg.setWindowTitle("Filter Column to Specific Value(s)")
        if dlg.exec():
            try:
                vals = dlg.getInputs()
                self.grades.filterColumnToValues(self.columnSelected, vals)
                self.reload()
                print('Values in '+str(self.columnSelected) +' filtered to: ' + str(vals))
            except Exception as e: 
                print(e)
                return

    def gpaMeanCol(self):
        try:
            self.grades.getGPAMeans()
            self.reload()
            print('Class grade standard deviations available in ' + self.grades.GPA_MEAN_COLUMN + ' column.')
        except Exception as e: 
            print(e)
            return

    def gpaDevCol(self):
        try:
            self.grades.getGPADeviations()
            self.reload()
            print('Class grade standard deviations available in ' + self.grades.GPA_STDDEV_COLUMN + ' column.')
        except Exception as e: 
            print(e)
            return

    def normCol(self):
        try:
            self.grades.getNormalizationColumn()
            self.reload()
            print('Normalized grades ((student grade - class mean) / stdDev) available in ' + self.grades.NORMALIZATION_COLUMN + ' column.')
        except Exception as e: 
            print(e)
            return

    def filterGPADeviation(self):
        dlg = filterGPADeviationDialogue(self)
        dlg.setWindowTitle("Filter Classes by GPA Deviation")
        if dlg.exec():
            try:
                # self.model.layoutAboutToBeChanged.emit()
                dev, out, fileNm = dlg.getInputs()
                self.grades.filterByGpaDeviationMoreThan(dev, out, fileNm + '.csv')
                self.reload()
                # self.model.layoutChanged.emit()
            except Exception as e: 
                print(e)
                return
            # print('Filtered data successfully.')

    @QtCore.pyqtSlot()
    def filterByStudentMajor(self):
        dlg = filterStudentMajorDialog(self)
        dlg.setWindowTitle("Filter to Classes and/or Departments")
        if dlg.exec():
            try:
                # self.model.layoutAboutToBeChanged.emit()
                majors = dlg.getInputs()
                self.grades.filterStudentsByMajors(majors)
                self.reload()
                # self.model.layoutChanged.emit()
                print('Majors filtered to: ' + str(majors))
            except Exception as e: 
                print(e)
                return
            print('Filtered data successfully.')

    @QtCore.pyqtSlot()
    def filterByClassOrMajorCorr(self):
        dlg = filterClassesDeptsDialog(self, True)
        dlg.setWindowTitle("Filter to Classes and/or Departments")
        if dlg.exec():
            try:
                # self.model.layoutAboutToBeChanged.emit()
                courses, depts, twoWay = dlg.getInputs()
                self.grades.filterToMultipleMajorsOrClasses(depts, courses, twoWay)
                self.reload()
                # self.model.layoutChanged.emit()
                print('Classes filtered to: ' + str(courses))
                print('Departments filtered to: ' + str(depts))
            except Exception as e: 
                print(e)
                return
            print('Filtered data successfully.')

    @QtCore.pyqtSlot()
    def filterByClassOrMajor(self):
        dlg = filterClassesDeptsDialog(self)
        dlg.setWindowTitle("Filter to Classes and/or Departments")
        if dlg.exec():
            try:
                # self.model.layoutAboutToBeChanged.emit()
                courses, depts = dlg.getInputs()
                self.grades.filterToMultipleMajorsOrClasses(depts, courses)
                # worker = Worker(self.grades.filterToMultipleMajorsOrClasses, majors=depts, classes=courses)
                # worker.signals.finished.connect(self.filterDone)
                # self.threadpool.start(worker)
                # self.model = TableModel(self.grades.df)
                self.reload()
                # self.model.layoutChanged.emit()
                print('Classes filtered to: ' + str(courses))
                print('Departments filtered to: ' + str(depts))
                print('Filtered data successfully.')
            except Exception as e: 
                print(e)
                pass

    def filterDone(self):
        self.model.layoutAboutToBeChanged.emit()
        self.model.layoutChanged.emit()
        print('Filtered to specific classes / departments.')

    def makeSankeyTrackNew(self):
        dlg = sankeyTrackInputNew(self)
        dlg.setWindowTitle("Export Class Track Graph")
        if dlg.exec():
            try:
                grTitle, fileTitle, group, required, minEdge = dlg.getInputs()
                worker = Worker(self.grades.sankeyGraphByCourseTracksOneGroup, courseGroup=group, requiredCourses=required, graphTitle=grTitle, outputName=fileTitle, minEdgeValue=minEdge)
                self.threadpool.start(worker)
            except Exception as e: 
                print(e)
                pass

    def makeSankeyTrack(self):
        dlg = sankeyTrackInput(self)
        dlg.setWindowTitle("Export Class Track Graph (alternate)")
        if dlg.exec():
            try:
                if dlg.orderedCheck.isChecked():
                  grTitle, fileTitle, minEdge, groups, thres = dlg.getInputs()
                  worker = Worker(self.grades.sankeyGraphByCourseTracks, courseGroups=groups, graphTitle=grTitle, outputName=fileTitle, minEdgeValue=minEdge, termThreshold=thres)
                else:
                  grTitle, fileTitle, minEdge, groups = dlg.getInputs()
                  worker = Worker(self.grades.sankeyGraphByCourseTracks, courseGroups=groups, graphTitle=grTitle, outputName=fileTitle, minEdgeValue=minEdge)
                self.threadpool.start(worker)
            except Exception as e: 
                print(e)
                pass

    def makeSankeyTrackAdvanced(self):
        # if 'termNumber' not in self.grades.columns:
        dlg = termNumberInput(self)
        dlg.setWindowTitle("Set Term Numbers / Ordering")
        if dlg.exec():
            try:
                termToVals = dlg.getInputs()
                print(termToVals)
                self.grades.termMapping(termToVals)
                self.reload()
                self.makeSankeyTrack()
            except Exception as e: 
                print(e)
                pass

    def exportMajorChord(self):
        dlg = majorChordInput(self)
        dlg.setWindowTitle("Export Major Chord Graph")
        if dlg.exec():
            try:
              corr, pval, name = dlg.getInputs()
              worker = Worker(self.grades.chordGraphByMajor, corr, pval, name, 200, 230, True, False)
              # worker.signals.finished.connect(self.chordDone(name))
              self.threadpool.start(worker) 
            except Exception as e: 
              print(e)
              pass
    def chordDone(self, name):
      def done():
        self.pic = PaintPicture(self, name + '.png')
      return done

    def exportCliqueHisto(self):
        dlg = cliqueHistogramInput(self)
        dlg.setWindowTitle("Export Clique Histogram")
        if dlg.exec():
            try:
              corr, dup, logSc, title, name = dlg.getInputs()
              worker = Worker(self.grades.outputCliqueDistribution, minCorr=corr, countDuplicates=dup, makeHistogram=True, fileName=name, graphTitle=title, logScale=logSc)
              # worker.signals.finished.connect(self.cliqueDone(name))
              self.threadpool.start(worker) 
            except Exception as e: 
              print(e)
              pass
    def cliqueDone(self, name):
      def done():
        self.pic = PaintPicture(self, name + '.png')
        # file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), name + ".html"))
        # self.web = QWebEngineView()
        # self.web.load(QUrl.fromLocalFile(file_path))
        # self.web.show()
      return done
    
    @QtCore.pyqtSlot()
    def instructorEffectiveness(self):
        dlg = instructorEffectivenessDialog(self)
        dlg.setWindowTitle("Rank Instructor Effectiveness")
        if dlg.exec():
            try:
                classOne, classTwo, filename, minStud = dlg.getInputs()
                if not filename.endswith('.csv'):
                    filename = "".join((filename, '.csv'))
                self.grades.instructorRanks(classOne,classTwo,fileName = filename, minStudents = minStud)
                print('Instructors of ' + classOne + ' class ranked based on future student performance in ' + classTwo + '. Saved to ' + filename + '.')
                dlg2 = csvPreview(self, filename)
                dlg2.setWindowTitle("Instructor Ranking - " + filename)
                dlg2.tableView.sortByColumn(1,1)
                # print (inspect.getsource(QtWidgets.QTableView.setSortingEnabled))
                # print (inspect.getsource(QtWidgets.QTableView.sortByColumn))
                dlg2.exec()
            except Exception as e: 
              print(e)
              pass

    @QtCore.pyqtSlot()
    def instructorEffectivenessAll(self):
        dlg = instructorEffectivenessDialog(self, True)
        dlg.setWindowTitle("Rank Instructors (All classes, may take hours)")
        if dlg.exec():
            try:
                filename, minStud, direct = dlg.getInputs()
                if not filename.endswith('.csv'):
                    filename = "".join((filename, '.csv'))
                worker = Worker(self.grades.instructorRanksAllClasses,fileName = filename, minStudents = minStud, directionality = direct)
                worker.signals.finished.connect(self.instructorEffectivenessAllProc(filename))
                self.threadpool.start(worker)
            except Exception as e: 
              print(e)
              pass
    def instructorEffectivenessAllProc(self, filename):
        def proc():
            print('Instructor rankings saved to ' + filename + '.')
            dlg2 = csvPreview(self, filename)
            dlg2.setWindowTitle("Instructor Ranking - " + filename)
            dlg2.tableView.sortByColumn(1,1)
            dlg2.exec()
        return proc
    @QtCore.pyqtSlot()
    def dictReplace(self):
        dlg = valuesReplaceDialogue(self)
        dlg.setWindowTitle("Substitute Values in Column")
        if dlg.exec():
            try:
                self.model.layoutAboutToBeChanged.emit()
                replace = dlg.getInputs()
                self.grades.df[str(self.columnSelected)] = self.grades.df[str(self.columnSelected)].map(replace).fillna(self.grades.df[str(self.columnSelected)])
                self.model.layoutChanged.emit()
                print('Replaced ' + str(len(replace)) + ' values in '+ str(self.columnSelected) +' column.')
                # self.reload()
            except Exception as e: 
              print(e)
              pass

    def dictReplaceFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Open", "","Text Files (*.txt)", options=options)
        if fileName:
            try:
                fh = open(fileName)
                lines = [line.rstrip('\n') for line in fh.readlines()]
                replace = {}
                for line in lines:
                  original, replacement = line.split(';')
                  replace[original] = replacement
                fh.close()
                self.model.layoutAboutToBeChanged.emit()
                self.grades.df[str(self.columnSelected)] = self.grades.df[str(self.columnSelected)].map(replace).fillna(self.grades.df[str(self.columnSelected)])
                self.model.layoutChanged.emit()
                print('Replaced ' + str(len(replace)) + ' values in '+ str(self.columnSelected) +' column.')
            except Exception as e: 
                print(e)
                pass

    @QtCore.pyqtSlot()
    def strReplace(self):
        dlg = substituteInput(self)
        dlg.setWindowTitle("Substitute String in Column " + str(self.columnSelected))
        if dlg.exec():
            try:
                self.model.layoutAboutToBeChanged.emit()
                subStr, replace = dlg.getInputs()
                self.grades.substituteSubStrInColumn(str(self.columnSelected), subStr, replace)
                self.model.layoutChanged.emit()
                print('Replaced ' + subStr + ' with ' + replace + ' in ' + str(self.columnSelected) + ' column.')
            except Exception as e: 
              print(e)
              pass

    @QtCore.pyqtSlot()
    def delColumn(self):
        try:
          if self.correlationFile and self.columnSelected in ['course1', 'course2', 'corr', 'P-value', '#students']:
              print('Error: column required for correlation calculations. Not deleted.')
          else:
              self.model.layoutAboutToBeChanged.emit()
              worker = Worker(self.grades.df.drop, self.columnSelected, axis=1, inplace=True)
              worker.signals.finished.connect(self.delColumnProc(self.columnSelected))
              self.threadpool.start(worker) 
        except Exception as e: 
          print(e)
          pass
    def delColumnProc(self, col):
        def proc():
            self.model.layoutChanged.emit()
            print('Removed ' + col + ' column.')
        return proc

    def write(self, text):
        self.terminal.write(text)
        text = text.strip()
        if len(text) > 0:
            self.message.setText(str(text))
    def flush(self):
        pass

    def originalReload(self):
      try:
        self.loadCsv(self.settings.value("lastFile", None))
        self.getLastKnownColumns()
        self.doColumnThings()
        self.setGeneralButtons()
        print('File reloaded.')
      except Exception as e: 
          print(e)
          print('Failed to reload.')
          pass
    
    def loadCsv(self, fileName):
        try:
            self.df = pd.read_csv(fileName, dtype=str)
            if {'course1', 'course2', 'corr', 'P-value', '#students'}.issubset(self.df.columns):
              self.correlationFile = True
              self.grades = classCorrelationData(self.df)
            else:
              self.correlationFile = False
              self.grades = gradeData(self.df)
              self.getLastKnownColumns()
            self.model = TableModel(self.grades.df)
            self.tableView.setModel(self.model)
            
            self.settings.setValue("lastFile", fileName)
            self.setWindowTitle("EDM - " + self.settings.value("lastFile",None))
        except: 
            pass
        
    def writeCsv(self, fileName):
        self.grades.exportCSV(fileName)

    def closeEvent(self, e):
        # Write window size and position to config file
        self.settings.setValue("size", self.size())
        self.settings.setValue("pos", self.pos())

        e.accept()


    @QtCore.pyqtSlot()
    def exportGPADistribution(self):
        dlg = gpaDistributionInput(self)
        dlg.setWindowTitle("Export GPA Distribution Histogram")
        if dlg.exec():
            try:
                graphTtl, fileNm, minClsses = dlg.getInputs()
                worker = Worker(self.grades.outputGpaDistribution, makeHistogram=True, fileName=fileNm, graphTitle=graphTtl, minClasses=minClsses)
                self.threadpool.start(worker) 
            except Exception as e: 
              print(e)
              pass
    def gpaDone(self, name):
      def done():
        self.pic = PaintPicture(self, name)
      return done

    @QtCore.pyqtSlot()
    def exportCorrelations(self, arg = None):
        dlg = getCorrDialogue(self)
        dlg.setWindowTitle("Export Class Correlations (Warning: may take hours)")
        if dlg.exec():
            try:
                minStdnts, fileNm, directedData, details = dlg.getInputs()
                worker = Worker(self.grades.exportCorrelationsWithMinNSharedStudents,filename=fileNm, nStudents=minStdnts, directedCorr = directedData, detailed = details)
                self.threadpool.start(worker)
            except Exception as e: 
              print(e)
              pass

    @QtCore.pyqtSlot()
    def designateGrades(self, arg = None):
      if self.grades and arg:
          if arg in self.grades.df.columns:
              self.grades.FINAL_GRADE_COLUMN = arg
              self.settings.setValue("Grades", arg)
          elif arg == ' ':
            self.grades.FINAL_GRADE_COLUMN = 'finalGrade'
            self.settings.setValue("Grades", None)
      elif self.grades and self.columnSelected:
          self.grades.FINAL_GRADE_COLUMN = self.columnSelected
          self.settings.setValue("Grades", self.columnSelected)
          print('Student Grade column set to ' + self.columnSelected)

    @QtCore.pyqtSlot()
    def designateCID(self, arg = None):
      if self.grades and arg:
          if arg in self.grades.df.columns:
              self.grades.CLASS_ID_COLUMN = arg
              self.settings.setValue("ClassID", arg)
          elif arg == ' ':
            self.grades.CLASS_ID_COLUMN = 'classID'
            self.settings.setValue("ClassID", None)
      elif self.grades and self.columnSelected:
          self.grades.CLASS_ID_COLUMN = self.columnSelected
          self.settings.setValue("ClassID", self.columnSelected)
          print('Class ID column set to ' + self.columnSelected)

    @QtCore.pyqtSlot()
    def designateFID(self, arg = None):
      if self.grades and arg:
          if arg in self.grades.df.columns:
              self.grades.FACULTY_ID_COLUMN = arg
              self.settings.setValue("FID", arg)
          elif arg == ' ':
            self.grades.FACULTY_ID_COLUMN = 'facultyID'
            self.settings.setValue("FID", None)
      elif self.grades and self.columnSelected:
          self.grades.FACULTY_ID_COLUMN = self.columnSelected
          self.settings.setValue("FID", self.columnSelected)
          print('Faculty ID column set to ' + self.columnSelected)

    @QtCore.pyqtSlot()
    def designateClssNmbr(self, arg = None):
      if self.grades and arg:
          if arg in self.grades.df.columns:
              self.grades.CLASS_NUMBER_COLUMN = arg
              self.settings.setValue("ClassNumber", arg)
          elif arg == ' ':
            self.grades.CLASS_CREDITS_COLUMN = 'classNumber'
            self.settings.setValue("ClassNumber", None)
      elif self.grades and self.columnSelected:
          self.grades.CLASS_NUMBER_COLUMN = self.columnSelected
          self.settings.setValue("ClassNumber", self.columnSelected)
          print('Class Number column set to ' + self.columnSelected)

    @QtCore.pyqtSlot()
    def designateCredits(self, arg = None):
      if self.grades and arg:
          if arg in self.grades.df.columns:
              self.grades.CLASS_CREDITS_COLUMN = arg
              self.settings.setValue("Credits", arg)
          elif arg == ' ':
            self.grades.CLASS_CREDITS_COLUMN = 'classCredits'
            self.settings.setValue("Credits", None)
      elif self.grades and self.columnSelected:
          self.grades.CLASS_CREDITS_COLUMN = self.columnSelected
          self.settings.setValue("Credits", self.columnSelected)
          print('Class Credits column set to ' + self.columnSelected)

    @QtCore.pyqtSlot()
    def designateDept(self, arg = None):
      if self.grades and arg:
          if arg in self.grades.df.columns:
              self.grades.CLASS_DEPT_COLUMN = arg
              self.settings.setValue("ClassDept", arg)
          elif arg == ' ':
            self.grades.CLASS_DEPT_COLUMN = 'classDept'
            self.settings.setValue("ClassDept", None)
      elif self.grades and self.columnSelected:
          self.grades.CLASS_DEPT_COLUMN = self.columnSelected
          self.settings.setValue("ClassDept", self.columnSelected)
          print('Class Department column set to ' + self.columnSelected)

    @QtCore.pyqtSlot()
    def designateSID(self, arg = None):
      if self.grades and arg:
          if arg in self.grades.df.columns:
              self.grades.STUDENT_ID_COLUMN = arg
              self.settings.setValue("StudentID", arg)
          elif arg == ' ':
            self.grades.STUDENT_ID_COLUMN = 'studentID'
            self.settings.setValue("StudentID", None)
      elif self.grades and self.columnSelected:
          self.grades.STUDENT_ID_COLUMN = self.columnSelected
          self.settings.setValue("StudentID", self.columnSelected)
          print('Student ID column set to ' + self.columnSelected)


    @QtCore.pyqtSlot()
    def designateStdntMjr(self, arg = None):
      if self.grades and arg:
          if arg in self.grades.df.columns:
              self.grades.STUDENT_MAJOR_COLUMN = arg
              self.settings.setValue("StudentMajor", arg)
          elif arg == ' ':
            self.grades.STUDENT_MAJOR_COLUMN = 'studentMajor'
            self.settings.setValue("StudentMajor", None)
      elif self.grades and self.columnSelected:
          self.grades.STUDENT_MAJOR_COLUMN = self.columnSelected
          self.settings.setValue("StudentMajor", self.columnSelected)
          print('Student Major column set to ' + self.columnSelected)

    @QtCore.pyqtSlot()
    def designateClassCode(self, arg = None):
      if self.grades and arg:
          if arg in self.grades.df.columns:
              self.grades.CLASS_CODE_COLUMN = arg
              self.settings.setValue("ClassCode", arg)
          elif arg == ' ':
            self.grades.CLASS_CODE_COLUMN = 'classCode'
            self.settings.setValue("ClassCode", None)
      elif self.grades and self.columnSelected:
          self.grades.CLASS_CODE_COLUMN = self.columnSelected
          self.settings.setValue("ClassCode", self.columnSelected)
          print('Class Code column set to ' + self.columnSelected)


    @QtCore.pyqtSlot()
    def designateTerms(self, arg = None):
      if self.grades and arg:
          if arg in self.grades.df.columns:
              self.grades.TERM_COLUMN = arg
              self.settings.setValue("Terms", arg)
          elif arg == ' ':
            self.grades.TERM_COLUMN = 'term'
            self.settings.setValue("Terms", None)
      elif self.grades and self.columnSelected:
          self.grades.TERM_COLUMN = self.columnSelected
          self.settings.setValue("Terms", self.columnSelected)
          print('Term column set to ' + self.columnSelected)

    @QtCore.pyqtSlot()
    def on_pushButtonWrite_clicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"Save As","","CSV Files (*.csv)", options=options)
        if fileName:
            if not fileName.endswith('.csv'):
              fileName = fileName + '.csv'
            self.writeCsv(fileName)
            print('File saved.')

    @QtCore.pyqtSlot()
    def on_pushButtonLoad_clicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Open", "","CSV Files (*.csv)", options=options)
        if fileName:
            self.loadCsv(fileName)
            self.getLastKnownColumns()
            self.doColumnThings()
            self.setGeneralButtons()
            print('File opened.')

class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data


    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = str(self._data.iloc[index.row(), index.column()])
            try:
              value = float(value)
              if value.is_integer():
                return str(int(value))
              return str(round(value, 3))
            except ValueError:
              return value

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]
    
    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Vertical:
                return str(self._data.index[section])
    def sort(self, column, order):
        """Sort table by given column number.
        """
        # print('sort clicked col {} order {}'.format(column, order))
        self.layoutAboutToBeChanged.emit()
        # print(self._df.columns[column])
        self._data[self._data.columns[column]] = pd.to_numeric(self._data[self._data.columns[column]],errors='ignore')
        self._data.sort_values(self._data.columns[column], ascending=order == Qt.AscendingOrder, inplace=True, kind='mergesort')
        self._data.reset_index(inplace=True, drop=True) # <-- this is the change
        # print(self._df)
        self.layoutChanged.emit()

class statsDialogue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        vlayout = QVBoxLayout()
        hLayout = QHBoxLayout()

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok, self)
        forms = []
        count = 0
        for column in parent.grades.df.columns.values:
          if count % 20 == 0:
            forms.append(QFormLayout())
            forms[-1].setVerticalSpacing(0)
          button = QPushButton(self)
          # self.buttons.append(button)
          vals = dict(parent.grades.df[column].value_counts())
          vals = {k: v for k, v in sorted(vals.items(), key=lambda item: item[1], reverse=True)}
          num = parent.grades.df[column].nunique()
          button.setText('Show Values')
          button.clicked.connect(self.showdialog(num, vals, column))
          forms[-1].addRow('Unique values in ' + str(column) + ': ' + str(num), button)
          count += 1
        for form in forms:
          hLayout.addLayout(form)
        vlayout.addLayout(hLayout)
        vlayout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        self.setLayout(vlayout)

    def showdialog(self, num, vals, column):
        def dlg():
            dlg2 = csvPreview(self, pd.DataFrame(vals.items(), columns=['Value', 'Frequency']))
            dlg2.resize(QSize(500, 480))
            dlg2.tableView.sortByColumn(1,1)
            dlg2.setWindowTitle("Stats on column - " + column)
            dlg2.exec()
            # msg = MyMessageBox()
            # # msg.setIcon(QMessageBox.Information)
            
            # if num < 500:
            #   msg.setInformativeText(str(vals)[1:-1])
            #   msg.setText(str(num) + " unique values found: ")
            # else:
            #   msg.setDetailedText(str(vals)[1:-1])
            #   msg.setText(str(num) + " unique values found:\t\t\t\t\t")
            # msg.setWindowTitle(str(column) +  " Unique Values")
            # # msg.setDetailedText(str(vals)[1:-1])
            # msg.setStandardButtons(QMessageBox.Ok)
            # msg.exec_()
        return dlg

    # def getInputs(self):
    #     return (self.first.value(), self.second.value(), self.third.text())

class valuesReplaceDialogue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        vlayout = QVBoxLayout()
        hLayout = QHBoxLayout()
        self.col = parent.columnSelected
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok, self)
        form = QFormLayout()
        form.setVerticalSpacing(0)
        self.inputs = []
        vals = sorted(list(parent.grades.df[self.col].astype(str).unique()))
        for val in vals:
          self.inputs.append((val, QLineEdit(self)))
          self.inputs[-1][1].setMaximumWidth(100)
          self.inputs[-1][1].setFixedWidth(100)
          form.addRow(str(val) + ' replacement: ', self.inputs[-1][1])
        hLayout.addLayout(form)
        scroll = QScrollArea() 
        widget = QWidget() 
        widget.setLayout(hLayout)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        vlayout.addWidget(scroll)
        form2 = QFormLayout()
        self.saveSubstitutions = QCheckBox()
        self.saveSubstitutions.setChecked(False)
        form2.addRow('Save substitutions to file ', self.saveSubstitutions)
        vlayout.addLayout(form2)
        vlayout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        self.setLayout(vlayout)

    def getInputs(self):
        replacements = {}
        for inputPair in self.inputs:
              if len(inputPair[1].text()) > 0:
                  replacements[inputPair[0]] = inputPair[1].text()
        if self.saveSubstitutions.isChecked() and len(replacements) > 0:
          with open(self.col+'_substitutions.txt', 'w') as f:
            for key, value in replacements.items():
              f.write(key + ';' + value + '\n')
        return (replacements)

class PaintPicture(QDialog):
    def __init__(self, parent=None, fileName=None):
        super(PaintPicture, self).__init__()

        layout = QVBoxLayout()
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok, self)

        image = QImage(fileName)

        self.imageLabel = QLabel()
        self.imageLabel.setPixmap(QPixmap.fromImage(image))

        layout.addWidget(self.imageLabel)
        layout.addWidget(buttonBox)

        self.setLayout(layout)
        self.setWindowTitle(fileName)
        self.show()
        buttonBox.accepted.connect(self.accept)



class MyMessageBox(QMessageBox):
    def __init__(self):
        self.isFirst = True
        QMessageBox.__init__(self)
        self.setSizeGripEnabled(True)

    def event(self, e):
        result = QMessageBox.event(self, e)

        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)
        self.setMinimumWidth(0)
        self.setMaximumWidth(16777215)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        textEdit = self.findChild(QTextEdit)
        if textEdit != None :
            textEdit.setMinimumHeight(0)
            textEdit.setMaximumHeight(16777215)
            textEdit.setMinimumWidth(0)
            textEdit.setMaximumWidth(16777215)
            textEdit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        if self.isFirst:
            button = self.findChild(QPushButton)
            if button != None:
                if button.text() == 'Show Details...':
                    button.clicked.emit()
                    self.isFirst = False

        return result

class columnDialog(QDialog):
    def __init__(self, parent=None, options = None):
        super().__init__(parent)
        def addOptions(combo):
          combo.addItem(' ')
          for option in options:
            combo.addItem(option)
        def setDefault(index, name):
          if parent.settings.value(name, None) in parent.grades.df.columns:
            self.combos[index].setCurrentIndex(parent.grades.df.columns.get_loc(parent.settings.value(name, None))+1)
        self.combos = []
        for i in range(10):
          self.combos.append(QComboBox(self))
          addOptions(self.combos[i])

        if parent.settings.value("ClassID", None):
          setDefault(2, "ClassID")
        if parent.settings.value("ClassCode", None):
          setDefault(9, "ClassCode")
        if parent.settings.value("FID", None):
          setDefault(8, "FID")
        if parent.settings.value("ClassNumber", None):
          setDefault(1, "ClassNumber")
        if parent.settings.value("ClassDept", None):
          setDefault(0, "ClassDept")
        if parent.settings.value("Grades", None):
          setDefault(4, "Grades")
        if parent.settings.value("StudentID", None):
          setDefault(3, "StudentID")
        if parent.settings.value("StudentMajor", None):
          setDefault(6, "StudentMajor")
        if parent.settings.value("Terms", None):
          setDefault(5, "Terms")
        if parent.settings.value("Credits", None):
          setDefault(7, "Credits")
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Class Department Column (Psych in Psych1000): ", self.combos[0])
        layout.addRow("Class Number Column (1000 in Psych1000): ", self.combos[1])
        layout.addRow("Class ID Column (number specific to class): ", self.combos[2])
        layout.addRow("Student ID Column: ", self.combos[3])
        layout.addRow("Student Grade Column (0.0 - 4.0+): ", self.combos[4])
        layout.addRow("Term Column (sortable): ", self.combos[5])
        layout.addRow("Student Major Column: ", self.combos[6])
        layout.addRow("Class Credits Column (optional): ", self.combos[7])
        layout.addRow("Faculty ID Column (optional): ", self.combos[8])
        layout.addRow("Class Code Column (optional): ", self.combos[9])
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.combos[0].currentText(),
                self.combos[1].currentText(),
                self.combos[2].currentText(),
                self.combos[3].currentText(),
                self.combos[4].currentText(),
                self.combos[5].currentText(),
                self.combos[6].currentText(),
                self.combos[7].currentText(),
                self.combos[8].currentText(),
                self.combos[9].currentText())

class getCorrDialogue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.first = QSpinBox(self)
        self.first.setMinimum(0)
        self.first.setSingleStep(1)
        self.first.setValue(20)
        self.second = QLineEdit(self)
        self.second.setText('classCorrelations')
        self.third = QCheckBox(self)
        self.third.setChecked(False)
        self.fourth = QCheckBox(self)
        self.fourth.setChecked(False)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Minimum number of students shared between two classes: ", self.first)
        layout.addRow("Output CSV file name: ", self.second)
        layout.addRow("Include class order correlations (much slower): ", self.third)
        layout.addRow("Include class grade details: ", self.fourth)

        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.first.value(), self.second.text(), self.third.isChecked(), self.fourth.isChecked())

class filterGPADeviationDialogue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.first = preciseSpinBox(self)
        self.first.setMinimum(0.0)
        self.first.setMaximum(1.0)
        self.first.setSingleStep(0.05)
        self.first.setValue(0.2)
        self.second = QCheckBox(self)
        self.second.setChecked(False)
        self.third = QLineEdit(self)
        self.third.setText('droppedData')
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Minimum class GPA Deviation: ", self.first)
        layout.addRow("Output dropped data to file: ", self.second)
        layout.addRow("Dropped data file name: ", self.third)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.first.value(), self.second.isChecked(), self.third.text())

class filterNumsDialogue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.firstBox = QGroupBox('Set Minimum')
        self.firstBox.setCheckable(True)
        self.secondBox = QGroupBox('Set Maximum')
        self.secondBox.setCheckable(True)
        # self.firstBox.setFlat(True)
        self.firstH = QHBoxLayout()
        self.secondH = QHBoxLayout()

        self.first = preciseSpinBox(self)
        # self.secondBox = QGroupBox('Minimum')
        self.second = preciseSpinBox(self)
        # self.firstCheck = QCheckBox(self)
        self.firstBox.setChecked(True)
        self.secondBox.setChecked(True)
        # self.secondCheck = QCheckBox(self)
        # self.secondCheck.setChecked(True)
        self.firstH.addWidget(self.first)
        self.secondH.addWidget(self.second)
        # self.firstH.addWidget(self.firstCheck)
        self.firstBox.setLayout(self.firstH)
        self.secondBox.setLayout(self.secondH)
        # self.firstBox.setStyleSheet("QGroupBox{padding-top:1px; margin-top:-20px}")
        # self.firstBox.setStyleSheet("border-radius: 0px; margin-top: 3ex;")
        # self.secondBox.setStyleSheet("border-radius:0px; margin-top:1ex;")

        # self.firstBox.setContentsMargins(0, 0, 0, 0)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QVBoxLayout(self)
        # layout.addRow("Minimum: ", self.firstBox)
        # layout.addRow("Apply Minimum: ", self.firstBox)
        layout.addWidget(self.firstBox)
        layout.addWidget(self.secondBox)
        # layout.addRow("Maximum: ", self.second)
        # layout.addRow("Apply Maximum: ", self.secondCheck)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        check = lambda x : x[0].value() if x[1].isChecked() else None           
        return (check([self.first, self.firstBox]), check([self.second, self.secondBox]))

class roundDialogue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.first = QSpinBox(self)
        self.first.setMinimum(0)
        self.first.setSingleStep(1)
        self.first.setValue(2)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Decimal places: ", self.first)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
            return (self.first.value())

class renameColumnDialogue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.first = QLineEdit(self)
        self.first.setMaximumWidth(200)
        self.first.setFixedWidth(200)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Rename"+ parent.columnSelected +" to: ", self.first)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
            return (self.first.text())

class filterValsDialogue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.first = QLineEdit(self)
        self.first.setMaximumWidth(400)
        self.first.setFixedWidth(400)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Value(s) (seperate by comma): ", self.first)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
            return (self.first.text().split(','))

class filterStudentMajorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.first = QLineEdit(self)
        self.first.setMaximumWidth(400)
        self.first.setFixedWidth(400)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Student Majors (seperate by comma): ", self.first)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
            return (self.first.text().split(','))
class csvPreview(QDialog):
    def __init__(self, parent = None, data=None):
        super().__init__(parent)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok, self)

        try:
            self.tableView = QtWidgets.QTableView(self)
            self.tableView.installEventFilter(self)

            if isinstance(data, str):
              self.df = pd.read_csv(data, dtype=str)
            elif isinstance(data, pd.DataFrame):
              self.df = data
            self.grades = gradeData(self.df)
            self.model = TableModel(self.grades.df)
            self.tableView.setModel(self.model)
            self.tableView.setSortingEnabled(True)
            self.tableView.horizontalHeader().setStretchLastSection(True)
            self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            self.tableView.horizontalHeader().setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.tableView.horizontalHeader().customContextMenuRequested.connect(self.onColumnRightClick)
            self.first = True
            self.doColumnThings()
        except: 
            pass

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.tableView)
        self.layout.addWidget(self.buttonBox)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.resize(QSize(700, 480))
        # self.setLayout(layout)
        # self.show()
    def onColumnRightClick(self, QPos=None):       
        parent=self.sender()
        pPos=parent.mapToGlobal(QtCore.QPoint(0, 0))
        mPos=pPos+QPos
        column = self.tableView.horizontalHeader().logicalIndexAt(QPos)
        label = self.model.headerData(column, Qt.Horizontal, Qt.DisplayRole)
        self.columnSelected = label
        self.rcMenu.move(mPos)
        self.rcMenu.show()

    def doColumnThings(self):
        self.columnSelected = None
        self.rcMenu=QMenu(self)
        self.substituteMenu = self.rcMenu.addMenu('Substitute')
        substituteInColumn = QAction('Substitute in Column...', self)
        substituteInColumn.triggered.connect(self.strReplace)
        self.substituteMenu.addAction(substituteInColumn)
        dictReplaceInColumn = QAction('Substitute Many Values...', self)
        dictReplaceInColumn.triggered.connect(self.dictReplace)
        self.substituteMenu.addAction(dictReplaceInColumn)
        dictReplaceTxt = QAction('Use substitution file...', self)
        dictReplaceTxt.triggered.connect(self.dictReplaceFile)
        self.substituteMenu.addAction(dictReplaceTxt)
        self.fNumMenu = self.rcMenu.addMenu('Filter / Numeric Operations')
        valFilter = QAction('Filter Column to Value(s)...', self)
        valFilter.triggered.connect(self.filterColByVals)
        self.fNumMenu.addAction(valFilter)
        NumericFilter = QAction('Filter Column Numerically...', self)
        NumericFilter.triggered.connect(self.filterColNumeric)
        self.fNumMenu.addAction(NumericFilter)
        absFilter = QAction('Make Absolute Values', self)
        absFilter.triggered.connect(self.absCol)
        self.fNumMenu.addAction(absFilter)
        avStats = QAction('Get Mean / Med. / Mode', self)
        avStats.triggered.connect(self.avCol)
        self.fNumMenu.addAction(avStats)
        roundFilter = QAction('Round Column...', self)
        roundFilter.triggered.connect(self.roundCol)
        self.fNumMenu.addAction(roundFilter)
        NAFilter = QAction('Drop Undefined Values in Column', self)
        NAFilter.triggered.connect(self.removeNaInColumn)
        self.rcMenu.addAction(NAFilter)
        # if not self.correlationFile:
        deleteColumn = QAction('Delete Column (permanent)', self)
        deleteColumn.triggered.connect(self.delColumn)
        self.rcMenu.addAction(deleteColumn)

    @QtCore.pyqtSlot()
    def strReplace(self):
        MyWindow.strReplace(self)
    @QtCore.pyqtSlot()
    def dictReplace(self):
        MyWindow.dictReplace(self)
    @QtCore.pyqtSlot()
    def dictReplaceFile(self):
        MyWindow.dictReplaceFile(self)
    @QtCore.pyqtSlot()
    def filterColByVals(self):
        MyWindow.filterColByVals(self)
    @QtCore.pyqtSlot()
    def filterColNumeric(self):
        MyWindow.filterColNumeric(self)
    @QtCore.pyqtSlot()
    def absCol(self):
        MyWindow.absCol(self)
    @QtCore.pyqtSlot()
    def avCol(self):
        MyWindow.avCol(self)  
    @QtCore.pyqtSlot()
    def roundCol(self):
        MyWindow.roundCol(self)  
    @QtCore.pyqtSlot()
    def removeNaInColumn(self):
        MyWindow.removeNaInColumn(self)
    @QtCore.pyqtSlot()
    def delColumn(self):
        MyWindow.delColumn(self)
    def reload(self):
        MyWindow.reload(self)
    def getLastKnownColumns(self):
        MyWindow.getLastKnownColumns(self)
    def eventFilter(self, source, event):
        if (event.type() == QtCore.QEvent.KeyPress and
            event.matches(QtGui.QKeySequence.Copy)):
            self.copySelection()
            return True
        return super(csvPreview, self).eventFilter(source, event)
    def copySelection(self):
        MyWindow.copySelection(self)

    def getInputs(self):
        return

class instructorEffectivenessDialog(QDialog):
    def __init__(self, parent=None, allClasses= False):
        super().__init__(parent)
        if not allClasses:
          self.first = QLineEdit(self)
          self.first.setMaximumWidth(200)
          self.first.setFixedWidth(200)
          self.second = QLineEdit(self)
          self.second.setMaximumWidth(200)
          self.second.setFixedWidth(200)
        self.third = QLineEdit(self)
        self.third.setMaximumWidth(200)
        self.third.setFixedWidth(200)
        self.third.setText('instructorRanking')
        self.fourth = QSpinBox(self)
        self.fourth.setMinimum(1)
        self.fourth.setMaximum(9999999999)
        self.fourth.setSingleStep(1)
        self.fourth.setValue(1)
        self.fourth.setMaximumWidth(200)
        self.fourth.setFixedWidth(200)
        if allClasses:
          self.fifth = preciseSpinBox(self)
          self.fifth.setMinimum(0.5)
          self.fifth.setMaximum(1.0)
          self.fifth.setSingleStep(0.1)
          self.fifth.setValue(0.8)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        if not allClasses:
          layout.addRow("First course (instructors from here): ", self.first)
          layout.addRow("Second course (indicates benefit of instructor): ", self.second)
        layout.addRow("Output file name: ", self.third)
        layout.addRow("Minimum number of Students per instructor: ", self.fourth)
        if allClasses:
          layout.addRow("Minimum Class Directionality (0.5 - 1.0): ", self.fifth)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        if not allClasses:
          return (self.first.text(), self.second.text(), self.third.text(), self.fourth.value())
        else:
          return (self.third.text(), self.fourth.value(), self.fifth.value())

class filterClassesDeptsDialog(QDialog):
    def __init__(self, parent=None, corr=False):
        super().__init__(parent)
        self.corrDialog = corr
        self.first = QLineEdit(self)
        self.first.setMaximumWidth(400)
        self.first.setFixedWidth(400)
        self.second = QLineEdit(self)
        self.second.setMaximumWidth(400)
        self.second.setFixedWidth(400)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Classes (dept+number or classcode, no spaces, sep. by comma): ", self.first)
        layout.addRow("Departments (seperate by comma): ", self.second)
        if corr:
          self.third = QCheckBox(self)
          self.third.setChecked(True)
          layout.addRow("Both classes must match requirements: ", self.third)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        if not self.corrDialog:
            return (self.first.text().split(','), self.second.text().split(','))
        else:
            return (self.first.text().split(','), self.second.text().split(','), self.third.isChecked())

class termNumberInput(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        vlayout = QVBoxLayout()
        hLayout = QHBoxLayout()

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        forms = []
        self.vals = []
        count = 0
        self.terms = sorted(list(parent.grades.df[parent.grades.TERM_COLUMN].unique()))
        for term in self.terms:
          if count % 15 == 0:
            forms.append(QFormLayout())
            forms[-1].setVerticalSpacing(0)
          self.vals.append(preciseSpinBox(dec=2))
          self.vals[-1].setMinimum(0.0)
          self.vals[-1].setSingleStep(0.05)
          self.vals[-1].setValue(round(count, 1))
          forms[-1].addRow(str(term) + ': ', self.vals[-1])
          count += 1
        for form in forms:
          hLayout.addLayout(form)
        # otherOpts = QFormLayout()
        
        vlayout.addLayout(hLayout)
        # vlayout.addLayout(otherOpts)
        vlayout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        self.setLayout(vlayout)

    def getInputs(self):
        termToVal = {}
        for i in range(len(self.terms)):
            termToVal[self.terms[i]] = round(self.vals[i].value(),2)
        return (termToVal)

class sankeyTrackInputNew(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.first = QLineEdit(self)
        self.first.setText('Class Tracks')
        self.second = QLineEdit(self)
        self.second.setText('sankeyTracks')
        self.third = QLineEdit(self)
        self.third.setMaximumWidth(300)
        self.third.setFixedWidth(300)
        self.fourth = QLineEdit(self)
        self.fourth.setMaximumWidth(300)
        self.fourth.setFixedWidth(300)
        self.fifth = QSpinBox(self)
        self.fifth.setMinimum(0)
        self.fifth.setSingleStep(1)
        self.fifth.setValue(0)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Graph Title: ", self.first)
        layout.addRow("File name: ", self.second)
        layout.addRow('Classes (seperate by comma): ', self.third)
        layout.addRow('Required Classes to Count Student (seperate by comma, all by default): ', self.fourth)
        layout.addRow('Minimum Edge Value: ', self.fifth)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.first.text(), self.second.text(), self.third.text().split(","), self.fourth.text().split(","), self.fifth.value())

class gradePredictDialogue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        parent.grades.getUniqueIdentifiersForSectionsAcrossTerms()
        self.classes = sorted(parent.grades.df[parent.grades.CLASS_CODE_COLUMN].unique().tolist())
        parent.grades.dropMissingValuesInColumn(parent.grades.FINAL_GRADE_COLUMN)
        parent.grades.convertColumnToNumeric(parent.grades.FINAL_GRADE_COLUMN)
        self.possibleGrades = [str(x) for x in sorted(parent.grades.df[parent.grades.FINAL_GRADE_COLUMN].unique().tolist(), reverse=True)]
        self.pastGradesBox = QGroupBox("Past Grades")
        self.form = QFormLayout()
        self.pastGradesCombos = []
        for x in range(5):
          self.addPastGrade()
        self.predictBox = QGroupBox("Classes to Predict")
        self.form2 = QFormLayout()
        self.predictCombos = []
        self.addPrediction()
        self.pastGradesBox.setLayout(self.form)
        self.predictBox.setLayout(self.form2)

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.pastGradesBox)
        self.plus = QPushButton(self)
        self.plus.setText("+")
        self.plus.clicked.connect(self.addPastGrade)
        self.minus = QPushButton(self)
        self.minus.setText("-")
        self.minus.clicked.connect(self.delPastGrade)
        firstButtons = QHBoxLayout()
        firstButtons.addWidget(self.minus)
        firstButtons.addWidget(self.plus)
        self.form.addRow(firstButtons)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.predictBox)
        self.plus2 = QPushButton(self)
        self.plus2.setText("+")
        self.plus2.clicked.connect(self.addPrediction)
        self.minus2 = QPushButton(self)
        self.minus2.setText("-")
        self.minus2.clicked.connect(self.delPrediction)
        secondButtons = QHBoxLayout()
        secondButtons.addWidget(self.minus2)
        secondButtons.addWidget(self.plus2)
        self.form2.addRow(secondButtons)

        self.modeBox = QGroupBox('Mode')
        self.modeLayout = QVBoxLayout()
        self.modes = ['Nearest Neighbor', 'Mean of Three Nearest']
        self.modeButtons = [QRadioButton(x) for x in self.modes]
        self.modeButtons[0].setChecked(True)
        for modeButton in self.modeButtons:
          self.modeLayout.addWidget(modeButton)
        self.modeBox.setLayout(self.modeLayout)
        self.vbox.addWidget(self.modeBox)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.vbox.addWidget(self.buttonBox)
        self.layout.addLayout(self.vbox)
        self.setLayout(self.layout)

    def addPastGrade(self):
        self.addGroup(self.form, self.pastGradesCombos, self.classes, self.possibleGrades)

    def delPastGrade(self):
        self.delGroup(self.form, self.pastGradesBox, self.pastGradesCombos)
      
    def addPrediction(self):
        self.predictCombos.append(QComboBox(self))
        self.predictCombos[-1].addItem(' ')
        self.fillCombo(self.predictCombos[-1], self.classes)
        self.form2.insertRow(len(self.predictCombos) - 1, 'Class ' + str(len(self.predictCombos)), self.predictCombos[-1])

    def delPrediction(self):
        self.delGroup(self.form2, self.predictBox, self.predictCombos)

    def addOptions(self, combos, options, options2):
        combos[0].addItem(' ')
        self.fillCombo(combos[0], options)
        self.fillCombo(combos[1], options2)

    def fillCombo(self, combo, optionList):
        for option in optionList:
          combo.addItem(option)

    def addGroup(self, form, combos, options, options2):
        combos.append((QComboBox(self), QComboBox(self)))
        self.addOptions(self.pastGradesCombos[-1], options, options2)
        form.insertRow(len(combos) - 1, combos[-1][0], combos[-1][1])

    def delGroup(self, form, group, combos):
        if len(combos) > 1:
            form.removeRow(form.rowCount()-2)
            del combos[-1]
            group.adjustSize()
            self.adjustSize()

    def getInputs(self):
        past = {x[0].currentText():float(x[1].currentText()) for x in self.pastGradesCombos if (x[0].currentText() != ' ')}
        predict = [x.currentText() for x in self.predictCombos if (x.currentText() != ' ')]
        mode = None
        translate = {'Nearest Neighbor':'nearest', 'Mean of Three Nearest':'nearestThree'}
        for modeButton in self.modeButtons:
          if modeButton.isChecked():
            mode = translate[modeButton.text()]

        return (past, predict, mode)

class sankeyTrackInput(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ordered = 'termOrder' in parent.grades.df.columns
        self.formGroupBox = QGroupBox("Sankey Graph")
        self.form = QFormLayout(self)
        self.first = QLineEdit(self)
        self.first.setText('Class Tracks')
        self.form.addRow('Graph Title: ', self.first)
        self.second = QLineEdit(self)
        self.second.setText('sankeyTracks')
        self.form.addRow('File Name: ', self.second)
        self.third = QSpinBox(self)
        self.third.setValue(0)
        self.form.addRow('Minimum Edge Value: ', self.third)
        if self.ordered:
          self.orderedCheck = QCheckBox(self)
          self.orderedCheck.setChecked(True)
          self.form.addRow('Use designated term order column: ', self.orderedCheck)
          self.maxConsecutive = preciseSpinBox(dec=2)
          self.maxConsecutive.setMinimum(0.0)
          self.maxConsecutive.setSingleStep(0.05)
          self.maxConsecutive.setValue(round(1,0))
          self.form.addRow('Maximum Difference for a consecutive term (if ordered): ', self.maxConsecutive)
        self.groups = []
        self.formGroupBox.setLayout(self.form)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.formGroupBox)
        self.addGroup()
        self.addGroup()
        self.plus = QPushButton(self)
        self.plus.setText("+")
        self.plus.clicked.connect(self.addGroup)
        self.minus = QPushButton(self)
        self.minus.setText("-")
        self.minus.clicked.connect(self.delGroup)
        self.layout.addWidget(self.plus)
        self.layout.addWidget(self.minus)
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def addGroup(self):
        line = QLineEdit()
        line.setMaximumWidth(300)
        line.setFixedWidth(300)
        self.groups.append(line)
        self.form.addRow('Class Group ' + str(len(self.groups)) + ' (seperate by commas only): ', self.groups[-1])

    def delGroup(self):
        if len(self.groups) > 2:
            self.form.removeRow(self.form.rowCount()-1)
            del self.groups[-1]
        self.formGroupBox.adjustSize()
        self.adjustSize()

    def getInputs(self):
        classGroups = [group.text().split(",") for group in self.groups]
        if self.orderedCheck.isChecked():
          return (self.first.text(), self.second.text(), self.third.value(), classGroups, self.maxConsecutive.value())
        return (self.first.text(), self.second.text(), self.third.value(), classGroups)

class gpaDistributionInput(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.first = QLineEdit(self)
        self.first.setText('GPA Distribution')
        self.second = QLineEdit(self)
        self.second.setText('gpaHistogram')
        self.third = QSpinBox(self)
        self.third.setMinimum(0)
        self.third.setSingleStep(1)
        self.third.setValue(36)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Graph Title: ", self.first)
        layout.addRow("File name: ", self.second)
        layout.addRow("Minimum number of classes taken to count GPA: ", self.third)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.first.text(), self.second.text(), self.third.value())

class cliqueHistogramInput(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.first = preciseSpinBox(self)
        self.first.setMinimum(0.0)
        self.first.setMaximum(1.0)
        self.first.setSingleStep(0.05)
        self.first.setValue(0.5)
        self.second = QCheckBox(self)
        self.third = QCheckBox(self)
        self.third.setChecked(True)
        self.fourth = QLineEdit(self)
        self.fourth.setText('Class Correlation Cliques')
        self.fifth = QLineEdit(self)
        self.fifth.setText('cliqueHistogram')
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Minimum correlation (0.0 to 1.0): ", self.first)
        layout.addRow("Count duplicate sub-cliques: ", self.second)
        layout.addRow("Y-axis in log base 10 scale: ", self.third)
        layout.addRow("Graph Title: ", self.fourth)
        layout.addRow("File name: ", self.fifth)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.first.value(), self.second.isChecked(), self.third.isChecked(), self.fourth.text(), self.fifth.text())

class preciseSpinBox(QDoubleSpinBox):
    def __init__(self, parent=None, dec = 15):
        super().__init__(parent)
        self.setFixedWidth(160)
        self.setDecimals(dec)
        self.setMaximum(999999999)
        self.setMinimum(-999999999)

    def textFromValue(self, val):
      return str(round(float(str(val)),self.decimals()))

class majorChordInput(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.first = preciseSpinBox(self)
        self.first.setMinimum(0.0)
        self.first.setMaximum(1.0)
        self.first.setSingleStep(0.05)
        self.first.setValue(0.5)
        self.second = preciseSpinBox(self)
        self.second.setMinimum(0.0)
        self.second.setMaximum(1.0)
        self.second.setSingleStep(0.05)
        self.second.setValue(0.05)
        self.third = QLineEdit(self)
        self.third.setText('majorGraph')
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Minimum correlation (0.0 to 1.0): ", self.first)
        layout.addRow("Maximum P-val (0.0 to 1.0): ", self.second)
        layout.addRow("File name: ", self.third)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.first.value(), self.second.value(), self.third.text())

class substituteInput(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.first = QLineEdit(self)
        self.second = QLineEdit(self)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Find: ", self.first)
        layout.addRow("Replacement: ", self.second)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.first.text(), self.second.text())

class TableWidgetCustom(QTableWidget):
    def __init__(self, parent=None):
        super(TableWidgetCustom, self).__init__(parent)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            self.copy()
        else:
            QTableWidget.keyPressEvent(self, event)

    def copy(self):
        selection = self.selectionModel()
        indexes = selection.selectedRows()
        if len(indexes) < 1:
            # No row selected
            return
        text = ''
        for idx in indexes:
            row = idx.row()
            for col in range(0, self.columnCount()):
                item = self.item(row, col)
                if item:
                    text += item.text()
                text += '\t'
            text += '\n'
        QApplication.clipboard().setText(text)

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data
    
    error
        `tuple` (exctype, value, traceback.format_exc() )
    
    result
        `object` data returned from processing, anything

    progress
        `int` indicating % progress 

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and 
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()    

        # Add the callback to our kwargs
        # self.kwargs['progress_callback'] = self.signals.progress        

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        
        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

if __name__ == "__main__":
    import sys
    import os
    # put it **before** importing webbroser
    os.environ["BROWSER"] = "firefox"
    import webbrowser
    # BROWSER = 'firefox'
    edmApplication = True
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('EDM Program')
    stylesAvailable = PyQt5.QtWidgets.QStyleFactory.keys()
    # print(stylesAvailable)
    if 'Macintosh' in stylesAvailable:
      app.setStyle('Macintosh')
    elif 'Breeze' in stylesAvailable:
      app.setStyle('Breeze')
    # elif 'GTK+' in stylesAvailable:
    #   app.setStyle('Breeze')
    elif 'Fusion' in stylesAvailable:
      app.setStyle('Fusion')
    elif 'Windows' in stylesAvailable:
      app.setStyle('Windows')

    main = MyWindow()
    main.show()

    sys.exit(app.exec_())