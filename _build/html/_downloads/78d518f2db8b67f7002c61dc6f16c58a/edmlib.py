"""
Library created by Michael Riad Zaky mriadzaky@fordham.edu (2020), with Daniel Leeds, Gary Weiss, Mavis Zhang at Fordham 
University. Library free for use provided you cite https://github.com/MichaelRZ/EDMLib in any resulting publications.  
Library free for redistribution provided you retain the author attributions above.

The following packages are required for installation before use: numpy, pandas, csv, scipy, holoviews
"""
import time
import numpy as np
import pandas as pd
import csv
import sys 
import math
from scipy.stats.stats import pearsonr
from scipy.stats import norm as sciNorm
import re, os
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

pvalSuffix = 'PValue'
ttestSuffix = '_ttest'
edmApplication = False

def makeExportDirectory(directory):
  global outDir
  if directory[-1] == '/':
    outDir = directory
  else:
    outDir = directory + '/'
  if not os.path.isdir(outDir[:-1]):
    os.mkdir(outDir[:-1])

outDir = 'exports/'  
makeExportDirectory(outDir)

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
  STUDENT_YEAR_COLUMN = 'studentYear'

  CLASS_ID_AND_TERM_COLUMN = 'courseIdAndTerm'
  CLASS_CODE_COLUMN = 'classCode'
  GPA_STDDEV_COLUMN = 'gpaStdDeviation'
  GPA_MEAN_COLUMN = 'gpaMean'
  NORMALIZATION_COLUMN = 'norm'
  GPA_NORMALIZATION_COLUMN = 'normByGpa'
  STUDENT_CLASS_NORMALIZATION_COLUMN = 'normByStudentByClass'

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

  def defineWorkingColumns(self, finalGrade, studentID, term, classID = 'classID', classDept = 'classDept', classNumber = 'classNumber', studentMajor = 'studentMajor', studentYear = 'studentYear', classCredits = 'classCredits', facultyID = 'facultyID', classCode = 'classCode'):
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
    self.STUDENT_YEAR_COLUMN = studentYear

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

  def dropNullAndConvertToNumeric(self, column):
    self.dropMissingValuesInColumn(column)
    self.convertColumnToNumeric(column)

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
      output_file(outDir +fileName + '.html', mode='inline')
      save(graph)
      show(graph)
      if not edmApplication:
        histo.opts(toolbar=None)
        graph = hv.render(histo)
        graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="10pt"), 'above')
        export_png(graph, filename=outDir +fileName + '.png')

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
    output_file(outDir +outputName + '.html', mode='inline')
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
        sortedClasses['termOrder'] = sortedClasses['termOrder'].apply(float)
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
    output_file(outDir +outputName + '.html', mode='inline')
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
      self.df[self.CLASS_CODE_COLUMN] = self.df[self.CLASS_DEPT_COLUMN].apply(str) + self.df[self.CLASS_NUMBER_COLUMN]
      self.df[self.CLASS_CODE_COLUMN] = self.df[self.CLASS_CODE_COLUMN].str.replace(" ","")
    if self.CLASS_ID_AND_TERM_COLUMN not in self.df.columns:
      if not self.__requiredColumnPresent(self.CLASS_ID_COLUMN):
        return
      if not self.__requiredColumnPresent(self.TERM_COLUMN):
        return
      self.df[self.CLASS_ID_AND_TERM_COLUMN] = self.df[self.CLASS_ID_COLUMN].apply(str) + self.df[self.TERM_COLUMN]
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

  def coursePairGraph(self, courseOne, courseTwo, fileName = 'coursePairGraph'):
    print('Creating course grade graph between ' + courseOne + ' and ' + courseTwo + '...')
    if not self.__requiredColumnPresent(self.STUDENT_ID_COLUMN):
      return
    if not self.__requiredColumnPresent(self.FINAL_GRADE_COLUMN):
      return
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
      if not self.__requiredColumnPresent(self.CLASS_CODE_COLUMN):
        return
    self.dropNullAndConvertToNumeric(self.FINAL_GRADE_COLUMN)
    classOneEntries = self.df.loc[self.df[self.CLASS_CODE_COLUMN] == courseOne]
    classTwoEntries = self.df.loc[self.df[self.CLASS_CODE_COLUMN] == courseTwo]
    studentsInClass = np.intersect1d(classOneEntries[self.STUDENT_ID_COLUMN].values,classTwoEntries[self.STUDENT_ID_COLUMN].values)
    classOneEntries.drop_duplicates(self.STUDENT_ID_COLUMN, keep='last', inplace=True)
    classTwoEntries.drop_duplicates(self.STUDENT_ID_COLUMN, keep='last', inplace=True)
    relevantEntriesOne = classOneEntries.loc[classOneEntries[self.STUDENT_ID_COLUMN].isin(studentsInClass)]
    relevantEntriesTwo = classTwoEntries.loc[classTwoEntries[self.STUDENT_ID_COLUMN].isin(studentsInClass)]
    relevantEntriesOne.sort_values(self.STUDENT_ID_COLUMN, inplace = True)
    relevantEntriesTwo.sort_values(self.STUDENT_ID_COLUMN, inplace = True)
    relevantEntriesOne.reset_index(inplace = True, drop=True)
    relevantEntriesTwo.reset_index(inplace = True, drop=True)
    relevantEntriesOne.rename(columns={self.FINAL_GRADE_COLUMN: courseOne}, inplace=True)
    relevantEntriesOne[courseTwo] = relevantEntriesTwo[self.FINAL_GRADE_COLUMN]
    combined = relevantEntriesOne.groupby([courseOne, courseTwo]).size().reset_index(name="freq")
    # print(combined)
    combined.to_csv(outDir+fileName+'.csv', index=False)
    # print(combined.values)
    n = combined['freq'].sum()
    combined['freq'] = combined['freq'] * (500 / n)
    points = hv.Points(combined.values, vdims=['frequency'])
    points.opts(opts.Points(size='frequency', xlabel=courseOne, ylabel=courseTwo, title=courseOne + ' Vs. ' + courseTwo + ' Grades'))
    hv.output(size=200)
    graph = hv.render(points)
    graph.add_layout(Title(text='n = ' + str(n), text_font_style="italic", text_font_size="10pt"), 'above')
    output_file(outDir + fileName + '.html', mode='inline')
    print('Exported ' + courseOne + ' and ' + courseTwo + ' grade graph to ' + fileName + '.html')
    save(graph)
    show(graph)
    hv.output(size=125)

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

    

  def instructorRanksAllClasses(self, fileName = 'completeInstructorRanks', minStudents = 20, directionality = 0.8, outputSubjectAverages = False, subjectFileName = 'instructorAverages', otherRank = None):
    """Create a table of instructors and their calculated benefit to students based on all classes they taught and future performance in all classes taken later. Exports a CSV file and returns a pandas dataframe.

      Args:
        fileName (:obj:`str`, optional): Name of CSV file to save. Set to 'completeInstructorRanks' by default.
        minStudents (:obj:`int`, optional): Minimum number of students to get data from for an instructor's entry to be included in the calculation. Set to 1 by default.
        directionality (:obj:`float`, optional): Minimum directionality (percentage of students who took one class before another). Range 0.0 to 1.0. Set to 0.8 by default.
        outputSubjectAverages (:obj:`bool`, optional): Output a file with averages of all the data in this file, by instructor, by subject. Set to :obj:`False` by default.
        subjectFileName (:obj:`str`, optional): File to output instructor/subject averages to. Set to 'instructorAverages' by default.

      Returns:
        :obj:`pandas.dataframe`: Pandas dataframe with columns indicating the instructor, the class taken, the future class, the normalized benefit to students, the grade point benefit to students, the number of students used to calculate for that instructor / class combination, as well as the number of students on the opposite side of that calculation (students in future class who did not take that instructor before).

    """
    if not self.__requiredColumnPresent(self.FACULTY_ID_COLUMN):
      return
    if self.NORMALIZATION_COLUMN not in self.df.columns:
      self.getNormalizationColumn()
    if not self.__requiredColumnPresent(self.NORMALIZATION_COLUMN):
      return
    self.dropNullAndConvertToNumeric(self.FINAL_GRADE_COLUMN)
    self.dropNullAndConvertToNumeric(self.NORMALIZATION_COLUMN)
    print('here')
    if directionality > 1.0 or directionality < 0.5:
      print('Error: directionality out of bounds (must be between 0.5 to 1, not '+ str(directionality) +').')
    if otherRank is not None:
      if not self.__requiredColumnPresent(otherRank):
        return
      self.dropNullAndConvertToNumeric(otherRank)
    print('here')
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
        newCount = sum(secondClassWithPastInstructor)
        nonStudents = len(secondClassWithPastInstructor.index) - newCount
        if nonStudents > 0:
          stdDev = secondClassEntries[self.FINAL_GRADE_COLUMN].std()
          if stdDev > 0:
            entriesWithPastInstructor = secondClassEntries.loc[secondClassWithPastInstructor]
            entriesWithoutPastInstructor = secondClassEntries.loc[~secondClassWithPastInstructor]
            AverageGradeWithInstructor = entriesWithPastInstructor[self.FINAL_GRADE_COLUMN].mean()
            AverageGradeWithoutInstructor = entriesWithoutPastInstructor[self.FINAL_GRADE_COLUMN].mean()
            rowDict = {}
            rowDict['Instructor'] = instructor
            rowDict['courseTaught'] = classOne
            rowDict['futureCourse'] = classTwo
            rowDict['normBenefit'] = entriesWithPastInstructor[self.NORMALIZATION_COLUMN].mean() - entriesWithoutPastInstructor[self.NORMALIZATION_COLUMN].mean()
            rowDict['gradeBenefit'] = (AverageGradeWithInstructor - AverageGradeWithoutInstructor) / stdDev
            if otherRank is not None:
              rowDict[otherRank] = entriesWithPastInstructor[otherRank].mean() - entriesWithoutPastInstructor[otherRank].mean()
            rowDict['#students'] = newCount
            rowDict['#nonStudents'] = nonStudents
            rowList.append(rowDict)
    print('here')
    classes = self.df[self.CLASS_CODE_COLUMN].unique().tolist()
    numClasses = len(classes)
    grouped = self.df.groupby(self.CLASS_CODE_COLUMN)
    for name, group in grouped:
      group.sort_values(self.TERM_COLUMN, inplace = True)
      group.drop_duplicates(self.STUDENT_ID_COLUMN, keep='last', inplace=True)
    for i in range(numClasses - 1):
      print('class ' + str(i+1) + '/' + str(numClasses))
      classOne = classes[i]
      oneDf = grouped.get_group(classOne)
      start_time = time.time()
      for j in range(i + 1, numClasses):
        classTwo = classes[j]
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
    if otherRank is None:
      completeDf = pd.DataFrame(rowList, columns=['Instructor','courseTaught','futureCourse','normBenefit','gradeBenefit','#students', '#nonStudents'])
    else:
      completeDf = pd.DataFrame(rowList, columns=['Instructor','courseTaught','futureCourse','normBenefit','gradeBenefit', otherRank, '#students', '#nonStudents'])
    completeDf.sort_values(by=['futureCourse','courseTaught','Instructor'])
    completeDf['Instructor'].replace(' ', np.nan, inplace=True)
    completeDf.dropna(subset=['Instructor'], inplace=True)
    completeDf.reset_index(inplace = True, drop=True)
    completeDf['totalStudents'] = (completeDf['#students'].apply(int)) + (completeDf['#nonStudents'].apply(int))
    completeDf['%ofStudents'] = ((completeDf['#students'].apply(float)) / (completeDf['totalStudents'].apply(float))) * 100
    completeDf['grade*Norm*Sign(norm)'] = completeDf['gradeBenefit'] * completeDf['normBenefit'] * np.sign(completeDf['normBenefit'])
    completeDf['normBenefit' + pvalSuffix] = pvalOfSeries(completeDf['normBenefit'])
    completeDf['gradeBenefit' + pvalSuffix] = pvalOfSeries(completeDf['gradeBenefit'])
    completeDf['grade*Norm*Sign(norm)' + pvalSuffix] = pvalOfSeries(completeDf['grade*Norm*Sign(norm)'])
    if otherRank is not None:
      completeDf[otherRank + pvalSuffix] = pvalOfSeries(completeDf[otherRank])

    if not fileName.endswith('.csv'):
      fileName = "".join((fileName, '.csv'))
    completeDf.to_csv(fileName, index=False)
    if outputSubjectAverages:
      instructorAveraging(completeDf, subjectFileName)
    return completeDf

  def getCorrelationsWithMinNSharedStudents(self, nSharedStudents = 20, directed = False, classDetails = False, sequenceDetails = False):
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
    nSharedStudents = max(nSharedStudents, 2)
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
    if sequenceDetails:
      if not self.__requiredColumnPresent(self.STUDENT_YEAR_COLUMN):
        return
      self.df[self.STUDENT_YEAR_COLUMN] = pd.to_numeric(self.df[self.STUDENT_YEAR_COLUMN],errors='ignore')

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
      if len(norms) < nSharedStudents and not sequenceDetails:
        return ([math.nan] * 36)
      elif len(norms) < nSharedStudents:
        return ([math.nan] * 52)
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
      AstdDevGrd = norms[self.FINAL_GRADE_COLUMN].std()
      BstdDevGrd = norms2[self.FINAL_GRADE_COLUMN].std()
      AstdDevNrm = norms[self.NORMALIZATION_COLUMN].std()
      BstdDevNrm = norms2[self.NORMALIZATION_COLUMN].std()
      ABASDGrd = aToBA[self.FINAL_GRADE_COLUMN].std()
      ABBSDGrd = aToBB[self.FINAL_GRADE_COLUMN].std()
      BAASDGrd = bToAA[self.FINAL_GRADE_COLUMN].std()
      BABSDGrd = bToAB[self.FINAL_GRADE_COLUMN].std()
      ABASDNrm = aToBA[self.NORMALIZATION_COLUMN].std()
      ABBSDNrm = aToBB[self.NORMALIZATION_COLUMN].std()
      BAASDNrm = bToAA[self.NORMALIZATION_COLUMN].std()
      BABSDNrm = bToAB[self.NORMALIZATION_COLUMN].std()
      AGrd = norms[self.FINAL_GRADE_COLUMN].mean()
      BGrd = norms2[self.FINAL_GRADE_COLUMN].mean()
      ANrm = norms[self.NORMALIZATION_COLUMN].mean()
      BNrm = norms2[self.NORMALIZATION_COLUMN].mean()
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
      
      if sequenceDetails:
        def yearTruths(x):
          return [numexpr.evaluate('(x == 1)'), numexpr.evaluate('(x == 2)'), 
                  numexpr.evaluate('(x == 3)'), numexpr.evaluate('(x == 4)')]
        c = aToBA[self.STUDENT_YEAR_COLUMN].values
        d = aToBB[self.STUDENT_YEAR_COLUMN].values
        e = bToAA[self.STUDENT_YEAR_COLUMN].values
        f = bToAB[self.STUDENT_YEAR_COLUMN].values

        aToBAFreshT, aToBASophT, aToBAJunT, aToBASenT = yearTruths(c)
        aToBBFreshT, aToBBSophT, aToBBJunT, aToBBSenT = yearTruths(d)
        bToAAFreshT, bToAASophT, bToAAJunT, bToAASenT = yearTruths(e)
        bToABFreshT, bToABSophT, bToABJunT, bToABSenT = yearTruths(f)
        crs1FreshMin = min(sum(aToBAFreshT), sum(bToAAFreshT))
        crs2FreshMin = min(sum(aToBBFreshT), sum(bToABFreshT))
        crs1SophMin = min(sum(aToBASophT), sum(bToAASophT))
        crs2SophMin = min(sum(aToBBSophT), sum(bToABSophT))
        crs1JunMin = min(sum(aToBAJunT), sum(bToAAJunT))
        crs2JunMin = min(sum(aToBBJunT), sum(bToABJunT))
        crs1SenMin = min(sum(aToBASenT), sum(bToAASenT))
        crs2SenMin = min(sum(aToBBSenT), sum(bToABSenT))
        aToBAFresh = aToBA.loc[aToBAFreshT]
        aToBASoph = aToBA.loc[aToBASophT]
        aToBAJun = aToBA.loc[aToBAJunT]
        aToBASen = aToBA.loc[aToBASenT]
        aToBBFresh = aToBB.loc[aToBBFreshT]
        aToBBSoph = aToBB.loc[aToBBSophT]
        aToBBJun = aToBB.loc[aToBBJunT]
        aToBBSen = aToBB.loc[aToBBSenT]
        bToAAFresh = bToAA.loc[bToAAFreshT]
        bToAASoph = bToAA.loc[bToAASophT]
        bToAAJun = bToAA.loc[bToAAJunT]
        bToAASen = bToAA.loc[bToAASenT]
        bToABFresh = bToAB.loc[bToABFreshT]
        bToABSoph = bToAB.loc[bToABSophT]
        bToABJun = bToAB.loc[bToABJunT]
        bToABSen = bToAB.loc[bToABSenT]
        nrmAlias, grdAlias = self.NORMALIZATION_COLUMN, self.FINAL_GRADE_COLUMN
        avNormDifCrs2Fresh = (aToBBFresh[nrmAlias].mean() - bToABFresh[nrmAlias].mean()) if crs2FreshMin > 0 else np.nan
        avNormDifCrs1Fresh = (aToBAFresh[nrmAlias].mean() - bToAAFresh[nrmAlias].mean()) if crs1FreshMin > 0 else np.nan
        avNormDifCrs2Soph = (aToBBSoph[nrmAlias].mean() - bToABSoph[nrmAlias].mean()) if crs2SophMin > 0 else np.nan
        avNormDifCrs1Soph = (aToBASoph[nrmAlias].mean() - bToAASoph[nrmAlias].mean()) if crs1SophMin > 0 else np.nan
        avNormDifCrs2Jun = (aToBBJun[nrmAlias].mean() - bToABJun[nrmAlias].mean()) if crs2JunMin > 0 else np.nan
        avNormDifCrs1Jun = (aToBAJun[nrmAlias].mean() - bToAAJun[nrmAlias].mean()) if crs1JunMin > 0 else np.nan
        avNormDifCrs2Sen = (aToBBSen[nrmAlias].mean() - bToABSen[nrmAlias].mean()) if crs2SenMin > 0 else np.nan
        avNormDifCrs1Sen = (aToBASen[nrmAlias].mean() - bToAASen[nrmAlias].mean()) if crs1SenMin > 0 else np.nan
        avGradeDifCrs2Fresh = (aToBBFresh[grdAlias].mean() - bToABFresh[grdAlias].mean()) if crs2FreshMin > 0 else np.nan
        avGradeDifCrs1Fresh = (aToBAFresh[grdAlias].mean() - bToAAFresh[grdAlias].mean()) if crs1FreshMin > 0 else np.nan
        avGradeDifCrs2Soph = (aToBBSoph[grdAlias].mean() - bToABSoph[grdAlias].mean()) if crs2SophMin > 0 else np.nan
        avGradeDifCrs1Soph = (aToBASoph[grdAlias].mean() - bToAASoph[grdAlias].mean()) if crs1SophMin > 0 else np.nan
        avGradeDifCrs2Jun = (aToBBJun[grdAlias].mean() - bToABJun[grdAlias].mean()) if crs2JunMin > 0 else np.nan
        avGradeDifCrs1Jun = (aToBAJun[grdAlias].mean() - bToAAJun[grdAlias].mean()) if crs1JunMin > 0 else np.nan
        avGradeDifCrs2Sen = (aToBBSen[grdAlias].mean() - bToABSen[grdAlias].mean()) if crs2SenMin > 0 else np.nan
        avGradeDifCrs1Sen = (aToBASen[grdAlias].mean() - bToAASen[grdAlias].mean()) if crs1SenMin > 0 else np.nan

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
      
      res = [corr, Pvalue, len(aNorms), corr1, Pvalue1, len(abANorms), corr2, Pvalue2, 
        len(baANorms), corr3, Pvalue3, len(concANorms), AGrd, BGrd, AstdDevGrd, BstdDevGrd, ANrm, BNrm, 
        AstdDevNrm, BstdDevNrm, ABASDGrd, ABBSDGrd, BAASDGrd, BABSDGrd, ABASDNrm, ABBSDNrm, BAASDNrm, BABSDNrm]

      if classDetails:
        res += [abAMean, abANormMean, abBMean, abBNormMean, baAMean, baANormMean, 
        baBMean, baBNormMean, concAMean, concANormMean, concBMean, concBNormMean]
      if sequenceDetails:
        res += [avNormDifCrs1Fresh, avNormDifCrs2Fresh, avNormDifCrs1Soph, avNormDifCrs2Soph, 
                avNormDifCrs1Jun, avNormDifCrs2Jun, avNormDifCrs1Sen, avNormDifCrs2Sen,
                avGradeDifCrs1Fresh, avGradeDifCrs2Fresh, avGradeDifCrs1Soph, avGradeDifCrs2Soph, 
                avGradeDifCrs1Jun, avGradeDifCrs2Jun, avGradeDifCrs1Sen, avGradeDifCrs2Sen,
                crs1FreshMin, crs2FreshMin, crs1SophMin, crs2SophMin, crs1JunMin, crs2JunMin, 
                crs1SenMin, crs2SenMin]
      return res

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
            r, p, c, r1, p1, c1, r2, p2, c2, r3, p3, c3, ag, bg, asg, bsg, an, bn, asn, bsn, abadevg, abbdevg, baadevg, babdevg, abadevn, abbdevn, baadevn, babdevn = result[:28]
            
            if not math.isnan(r):
              if classDetails and sequenceDetails:
                abA, abANorm, abB, abBNorm, baA, baANorm, baB, baBNorm, concA, concANorm, concB, concBNorm = result[28:40]
                avNormDifCrs1Fresh, avNormDifCrs2Fresh, avNormDifCrs1Soph, avNormDifCrs2Soph, avNormDifCrs1Jun, avNormDifCrs2Jun, avNormDifCrs1Sen, avNormDifCrs2Sen, avGradeDifCrs1Fresh, avGradeDifCrs2Fresh, avGradeDifCrs1Soph, avGradeDifCrs2Soph, avGradeDifCrs1Jun, avGradeDifCrs2Jun, avGradeDifCrs1Sen, avGradeDifCrs2Sen, crs1FreshMin, crs2FreshMin, crs1SophMin, crs2SophMin, crs1JunMin, crs2JunMin, crs1SenMin, crs2SenMin = result[40:]
                f.append((n, m, r, p, c, ag, bg, asg, bsg, an, bn, asn, bsn, abadevg, abbdevg, baadevg, babdevg, abadevn, abbdevn, baadevn, babdevn, r1, p1, c1, abA, abANorm, abB, abBNorm, r2, p2, c2, baB, baBNorm, baA, baANorm, r3, p3, c3, concA, concANorm, concB, concBNorm, avNormDifCrs1Fresh, avNormDifCrs2Fresh, avNormDifCrs1Soph, avNormDifCrs2Soph, avNormDifCrs1Jun, avNormDifCrs2Jun, avNormDifCrs1Sen, avNormDifCrs2Sen, avGradeDifCrs1Fresh, avGradeDifCrs2Fresh, avGradeDifCrs1Soph, avGradeDifCrs2Soph, avGradeDifCrs1Jun, avGradeDifCrs2Jun, avGradeDifCrs1Sen, avGradeDifCrs2Sen, crs1FreshMin, crs2FreshMin, crs1SophMin, crs2SophMin, crs1JunMin, crs2JunMin, crs1SenMin, crs2SenMin))
                if n != m:
                  f.append((m, n, r, p, c, bg, ag, bsg, asg, bn, an, bsn, asn, babdevg, baadevg, abbdevg, abadevg, babdevn, baadevn, abbdevn, abadevn,r2, p2, c2, baB, baBNorm, baA, baANorm, r1, p1, c1, abA, abANorm, abB, abBNorm, r3, p3, c3, concB, concBNorm, concA, concANorm, -avNormDifCrs2Fresh, -avNormDifCrs1Fresh, -avNormDifCrs2Soph, -avNormDifCrs1Soph, -avNormDifCrs2Jun, -avNormDifCrs1Jun, -avNormDifCrs2Sen, -avNormDifCrs1Sen, -avGradeDifCrs2Fresh, -avGradeDifCrs1Fresh, -avGradeDifCrs2Soph, -avGradeDifCrs1Soph, -avGradeDifCrs2Jun, -avGradeDifCrs1Jun, -avGradeDifCrs2Sen, -avGradeDifCrs1Sen, crs2FreshMin, crs1FreshMin, crs2SophMin, crs1SophMin, crs2JunMin, crs1JunMin, crs2SenMin, crs1SenMin))
              elif classDetails:
                abA, abANorm, abB, abBNorm, baA, baANorm, baB, baBNorm, concA, concANorm, concB, concBNorm = result[28:]
                # print(n + " " + m + " " + str(r))
                f.append((n, m, r, p, c, ag, bg, asg, bsg, an, bn, asn, bsn, abadevg, abbdevg, baadevg, babdevg, abadevn, abbdevn, baadevn, babdevn,r1, p1, c1, abA, abANorm, abB, abBNorm, r2, p2, c2, baB, baBNorm, baA, baANorm, r3, p3, c3, concA, concANorm, concB, concBNorm))
                if n != m:
                  f.append((m, n, r, p, c, bg, ag, bsg, asg, bn, an, bsn, asn, babdevg, baadevg, abbdevg, abadevg, babdevn, baadevn, abbdevn, abadevn, r2, p2, c2, baB, baBNorm, baA, baANorm, r1, p1, c1, abA, abANorm, abB, abBNorm, r3, p3, c3, concB, concBNorm, concA, concANorm))
              else:
                f.append((n, m, r, p, c, ag, bg, asg, bsg, an, bn, asn, bsn, r1, p1, c1, r2, p2, c2, r3, p3, c3))
                if n != m:
                  f.append((m, n, r, p, c, bg, ag, bsg, asg, bn, an, bsn, asn, r2, p2, c2, r1, p1, c1, r3, p3, c3))
      classesProcessed.add(n)
      print(str(time.time() - tim))
    f[:] = [x for x in f if isinstance(x[0], str)]
    f.sort(key = lambda x: x[1])
    f.sort(key = lambda x: x[0])
    if not directed:
      normoutput = pd.DataFrame(f, columns=('course1', 'course2', 'corr', 'P-value', '#students','avGrade1','avGrade2','stdDevGrade1','stdDevGrade2','avNorm1','avNorm2','stdDevNorm1','stdDevNorm2'))
    else:
      if not classDetails:
        normoutput = pd.DataFrame(f, columns=('course1', 'course2', 'corr', 'P-value', '#students','avGrade1','avGrade2','stdDevGrade1','stdDevGrade2','avNorm1','avNorm2','stdDevNorm1','stdDevNorm2', 'corrCourse1->2', 'P-valueCrs1->2','#studentsCrs1->2', 'corrCourse2->1', 'P-valueCrs2->1','#studentsCrs2->1', 'corrCoursesConcurrent', 'P-valueCrsConcurrent','#studentsCrsConcurrent', 'crs1->2grdStdDev(crs1)','crs1->2grdStdDev(crs2)','crs2->1grdStdDev(crs1)','crs2->1grdStdDev(crs2)','crs1->2nrmStdDev(crs1)','crs1->2nrmStdDev(crs2)','crs2->1nrmStdDev(crs1)','crs2->1nrmStdDev(crs2)'))
      else:
        if not sequenceDetails:
          normoutput = pd.DataFrame(f, columns=('course1', 'course2', 'corr', 'P-value', '#students','avGrade1','avGrade2','stdDevGrade1','stdDevGrade2','avNorm1','avNorm2','stdDevNorm1','stdDevNorm2', 'crs1->2grdStdDev(crs1)','crs1->2grdStdDev(crs2)','crs2->1grdStdDev(crs1)','crs2->1grdStdDev(crs2)','crs1->2nrmStdDev(crs1)','crs1->2nrmStdDev(crs2)','crs2->1nrmStdDev(crs1)','crs2->1nrmStdDev(crs2)', 'corrCourse1->2', 'P-valueCrs1->2','#studentsCrs1->2', 'Av.GradeCrs1->2(crs1)', 'Av.NormCrs1->2(crs1)', 'Av.GradeCrs1->2(crs2)','Av.NormCrs1->2(crs2)','corrCourse2->1', 'P-valueCrs2->1','#studentsCrs2->1', 'Av.GradeCrs2->1(crs2)','Av.NormCrs2->1(crs2)', 'Av.GradeCrs2->1(crs1)', 'Av.NormCrs2->1(crs1)', 'corrCoursesConcurrent', 'P-valueCrsConcurrent','#studentsCrsConcurrent','Av.GradeConcurrent(crs1)','Av.NormConcurrent(crs1)','Av.GradeConcurrent(crs2)','Av.NormConcurrent(crs2)'))
        else:
          normoutput = pd.DataFrame(f, columns=('course1', 'course2', 'corr', 'P-value', '#students','avGrade1','avGrade2','stdDevGrade1','stdDevGrade2','avNorm1','avNorm2','stdDevNorm1','stdDevNorm2', 'crs1->2grdStdDev(crs1)','crs1->2grdStdDev(crs2)','crs2->1grdStdDev(crs1)','crs2->1grdStdDev(crs2)','crs1->2nrmStdDev(crs1)','crs1->2nrmStdDev(crs2)','crs2->1nrmStdDev(crs1)','crs2->1nrmStdDev(crs2)','corrCourse1->2', 'P-valueCrs1->2','#studentsCrs1->2', 'Av.GradeCrs1->2(crs1)', 'Av.NormCrs1->2(crs1)', 'Av.GradeCrs1->2(crs2)','Av.NormCrs1->2(crs2)','corrCourse2->1', 'P-valueCrs2->1','#studentsCrs2->1', 'Av.GradeCrs2->1(crs2)','Av.NormCrs2->1(crs2)', 'Av.GradeCrs2->1(crs1)', 'Av.NormCrs2->1(crs1)', 'corrCoursesConcurrent', 'P-valueCrsConcurrent','#studentsCrsConcurrent','Av.GradeConcurrent(crs1)','Av.NormConcurrent(crs1)','Av.GradeConcurrent(crs2)','Av.NormConcurrent(crs2)', 'Av.NormCrs1->2-Av.NormCrs2->1(crs1_fresh)', 'Av.NormCrs1->2-Av.NormCrs2->1(crs2_fresh)', 'Av.NormCrs1->2-Av.NormCrs2->1(crs1_soph)', 'Av.NormCrs1->2-Av.NormCrs2->1(crs2_soph)', 'Av.NormCrs1->2-Av.NormCrs2->1(crs1_jun)', 'Av.NormCrs1->2-Av.NormCrs2->1(crs2_jun)', 'Av.NormCrs1->2-Av.NormCrs2->1(crs1_sen)', 'Av.NormCrs1->2-Av.NormCrs2->1(crs2_sen)', 'Av.GradeCrs1->2-Av.GradeCrs2->1(crs1_fresh)', 'Av.GradeCrs1->2-Av.GradeCrs2->1(crs2_fresh)', 'Av.GradeCrs1->2-Av.GradeCrs2->1(crs1_soph)', 'Av.GradeCrs1->2-Av.GradeCrs2->1(crs2_soph)', 'Av.GradeCrs1->2-Av.GradeCrs2->1(crs1_jun)', 'Av.GradeCrs1->2-Av.GradeCrs2->1(crs2_jun)', 'Av.GradeCrs1->2-Av.GradeCrs2->1(crs1_sen)', 'Av.GradeCrs1->2-Av.GradeCrs2->1(crs2_sen)', 'crs1FreshMin', 'crs2FreshMin', 'crs1SophMin', 'crs2SophMin', 'crs1JunMin', 'crs2JunMin', 'crs1SenMin', 'crs2SenMin'))
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

  def exportCorrelationsWithMinNSharedStudents(self, filename = 'CorrelationOutput_EDMLIB.csv', nStudents = 20, directedCorr = False, detailed = False, sequenced = False):
    """Exports CSV file with all correlations between classes with the given minimum number of shared students. File format has columns 'course1', 'course2', 'corr', 'P-value', '#students'.

    Args:
        fileName (:obj:`str`, optional): Name of CSV to output. Default 'CorrelationOutput_EDMLIB.csv'.
        nStudents (:obj:`int`, optional): Minimum number of shared students a pair of classes must have to compute a correlation. Defaults to 20.
        directedCorr (:obj:`bool`, optional): Whether or not to include data specific to students who took class A before B, vice versa, and concurrently. Defaults to 'False'.
        detailed (:obj:`bool`, optional): Whether or not to include means of student grades, normalized grades, and standard deviations used. Defaults to 'False'.

    """
    if not filename.endswith('.csv'):
      filename = "".join((filename, '.csv'))
    self.getCorrelationsWithMinNSharedStudents(nSharedStudents=nStudents, directed=directedCorr, classDetails = detailed, sequenceDetails = sequenced).to_csv(filename, index=False)

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

  def getDictOfStudentGPAs(self, getStdDev = False):
    self.dropNullAndConvertToNumeric(self.FINAL_GRADE_COLUMN)
    if self.CLASS_CREDITS_COLUMN in self.df.columns:
      self.dropNullAndConvertToNumeric(self.CLASS_CREDITS_COLUMN)
      temp = self.df.loc[:, [self.STUDENT_ID_COLUMN, self.CLASS_CREDITS_COLUMN, self.FINAL_GRADE_COLUMN]]
      temp['classPoints'] = temp[self.CLASS_CREDITS_COLUMN] * temp[self.FINAL_GRADE_COLUMN]
      temp2 = temp.groupby(self.STUDENT_ID_COLUMN, as_index=False)
      sums = temp2.sum()
      sums['gpa'] = sums['classPoints'] / sums[self.CLASS_CREDITS_COLUMN]
      # print(temp2)
      # print(sums)
      gpas = dict(zip(sums[self.STUDENT_ID_COLUMN], sums['gpa']))
      if getStdDev:
        # temp2.apply(lambda x: print(x))
        sums['stddev'] = temp2.apply(lambda x: np.average((x[self.FINAL_GRADE_COLUMN]-gpas[x[self.STUDENT_ID_COLUMN].iloc[0]])**2, weights=x[self.CLASS_CREDITS_COLUMN]))
        stdDevs = dict(zip(sums[self.STUDENT_ID_COLUMN], sums['stddev']))
        # np.average(subEntries['normBenefit'].astype(float), weights=subEntries[weighting].astype(float))
      # print(sums)
    else:
      temp = self.df.loc[:, [self.STUDENT_ID_COLUMN, self.FINAL_GRADE_COLUMN]]
      temp2 = temp.groupby(self.STUDENT_ID_COLUMN, as_index=False)
      means = temp2.mean()
      gpas = dict(zip(means[self.STUDENT_ID_COLUMN], means[self.FINAL_GRADE_COLUMN]))
      if getStdDev:
        gpas['stds'] = temp2.apply(lambda x: np.average((x[self.FINAL_GRADE_COLUMN]-gpas[x[self.STUDENT_ID_COLUMN].iloc[0]])**2))
        stdDevs = dict(zip(stds[self.STUDENT_ID_COLUMN], gpas['stds']))
    
    if getStdDev:
      return (gpas, stdDevs)
    return gpas

  def getNormalizationByGPA(self):
    if not self.__requiredColumnPresent(self.STUDENT_ID_COLUMN):
      return
    if not self.__requiredColumnPresent(self.FINAL_GRADE_COLUMN):
      return
    
    gpas, stds = self.getDictOfStudentGPAs(getStdDev=True)
    temp = self.df.loc[:,[self.STUDENT_ID_COLUMN, self.FINAL_GRADE_COLUMN]]
    def rowOp(row):
      try:
        res = (row[self.FINAL_GRADE_COLUMN] - gpas[row[self.STUDENT_ID_COLUMN]]) / stds[row[self.STUDENT_ID_COLUMN]]
        return res
      except ZeroDivisionError:
        return math.nan
    self.df[self.GPA_NORMALIZATION_COLUMN] = temp.apply(rowOp, axis = 1)

  def getNormalizationByStudentByClass(self):
    if not self.__requiredColumnPresent(self.STUDENT_ID_COLUMN):
      return
    if not self.__requiredColumnPresent(self.FINAL_GRADE_COLUMN):
      return
    if self.NORMALIZATION_COLUMN not in self.df.columns:
      self.getNormalizationColumn()
      if not self.__requiredColumnPresent(self.NORMALIZATION_COLUMN):
        return
    self.dropNullAndConvertToNumeric(self.NORMALIZATION_COLUMN)
    temp = self.df.loc[:,[self.STUDENT_ID_COLUMN, self.NORMALIZATION_COLUMN]]
    temp2 = temp.groupby(self.STUDENT_ID_COLUMN, as_index=False)
    means = temp2.mean()
    meanDict = dict(zip(means[self.STUDENT_ID_COLUMN], means[self.NORMALIZATION_COLUMN]))
    print(means)
    # print(meanDict)
    # print(temp2)
    stds = temp2.apply(lambda x: np.average((x[self.NORMALIZATION_COLUMN]-meanDict[x[self.STUDENT_ID_COLUMN].iloc[0]])**2))
    stds.columns = [self.NORMALIZATION_COLUMN, 'stds']
    means['stds'] = stds['stds']
    print(means)
    stdDict = dict(zip(means[self.STUDENT_ID_COLUMN], means['stds']))
    def rowOp(row):
      try:
        res = (row[self.NORMALIZATION_COLUMN] - meanDict[row[self.STUDENT_ID_COLUMN]]) / stdDict[row[self.STUDENT_ID_COLUMN]]
        return res
      except ZeroDivisionError:
        return math.nan
    self.df[self.STUDENT_CLASS_NORMALIZATION_COLUMN] = temp.apply(rowOp, axis = 1)

  def getGradesByYear(self):
    if self.NORMALIZATION_COLUMN not in self.df.columns:
      self.getNormalizationColumn()
    if self.NORMALIZATION_COLUMN not in self.df.columns:
      print('Error: Could not find normalization column.')
      return
    if not self.__requiredColumnPresent(self.STUDENT_YEAR_COLUMN):
      return
    years = sorted(self.df[self.STUDENT_YEAR_COLUMN].unique().tolist())
    classes = self.df[self.CLASS_CODE_COLUMN].unique().tolist()
    # grouped = self.df.groupby(self.STUDENT_YEAR_COLUMN)
    # temp = self.df.loc[:, [self.CLASS_ID_AND_TERM_COLUMN, self.FINAL_GRADE_COLUMN]]
    # meanGpas = temp.groupby(self.CLASS_ID_AND_TERM_COLUMN).mean()
    # meanGpas.rename(columns={self.FINAL_GRADE_COLUMN:self.GPA_MEAN_COLUMN}, inplace=True)
    # self.df = pd.merge(self.df,meanGpas, on=self.CLASS_ID_AND_TERM_COLUMN)
    dictList = []
    for year in years:
      yearGroup = self.df.loc[self.df[self.STUDENT_YEAR_COLUMN] == year]
      yearDict = {}
      print(year)
      counter = 0
      count = len(classes)
      for course in classes:
        counter += 1
        print(str(counter) + '/' + str(count) + ' classes')
        classYearGroup = yearGroup[self.CLASS_CODE_COLUMN] == course
        if any(classYearGroup):
          classGroup = yearGroup.loc[classYearGroup]
          yearDict[course] = classGroup[self.NORMALIZATION_COLUMN].mean()
        else:
          yearDict[course] = np.nan
      dictList.append(yearDict)

    def getMean(year, course):
      return dictList[years.index(year)][course]
    for item in years:
      self.df[item+'Norm'] = self.df.apply(lambda row: getMean(item, row[self.CLASS_CODE_COLUMN]), axis=1)


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
    self.dropNullAndConvertToNumeric(self.FINAL_GRADE_COLUMN)
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
    self.convertColumnToNumeric(self.GPA_STDDEV_COLUMN)
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
    if not fileName.endswith('.csv'):
      fileName = "".join((fileName, '.csv'))
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
    output_file(outDir +outputName + '.html', mode='inline')
    save(graph)
    if showGraph:
      show(graph)
    chord.opts(toolbar=None)
    if outputImage:
      hv.output(size=imageSize)
      export_png(hv.render(chord), filename=outDir +outputName + '.png')
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
      subtitle = 'n = ' + str(self.getEntryCount())
      if minCorr:
        subtitle = 'corr >= ' + str(minCorr) + ', ' + subtitle

      graph = hv.render(histo)
      graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="10pt"), 'above')
      output_file(outDir +fileName + '.html', mode='inline')
      save(graph)
      show(graph)
      if not edmApplication:
        histo.opts(toolbar=None)
        graph = hv.render(histo)
        graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="10pt"), 'above')
        export_png(graph, filename=outDir +fileName + '.png')

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

  def convertColumnToString(self, column):
    if not self.__requiredColumnPresent(column):
      return
    self.df.astype({column:str}, copy=False)
  
  def __requiredColumnPresent(self, column):
    if column not in self.df.columns:
      if edmApplication:
        print("Error: required column '" + column + "' not present in dataset. Fix by right clicking / setting columns.")
      else:
        print("Error: required column '" + column + "' not present in dataset. Fix or rename with the 'defineWorkingColumns' function.")
      return False
    return True

  

def instructorAveraging(data, filename = 'instructorAverages', minPercentage = 5.0, maxPercentage = 90.0, weighting = '#students', extraNorm = None):
  cols = ['Instructor','courseTaught','futureCourse','normBenefit','gradeBenefit','#students', '#nonStudents', 'totalStudents', '%ofStudents']
  if not all(x in data.columns for x in cols):
    print('Error: Columns missing. Did you use the instructorRanksAllClasses method?')
    return
  if weighting not in data.columns and weighting is not None:
    print('Error: Weight not present in columns. Check spelling.')
    return
  filteredData = data.loc[(data['courseTaught'].str.replace('\d+', '') == data['futureCourse'].str.replace('\d+', '')) & 
    (data['%ofStudents'].apply(float) <= maxPercentage) & (data['%ofStudents'].apply(float) >= minPercentage)]
  filteredData['courseTaught'] = data['courseTaught'].str.replace('\d+', '')
  filteredData['futureCourse'] = data['futureCourse'].str.replace('\d+', '')
  uniqueInstructors = filteredData['Instructor'].unique()
  grouped = filteredData.groupby('Instructor')
  rowlist = []
  for instructor in uniqueInstructors:
    entries = grouped.get_group(instructor)
    for subject in entries['courseTaught'].unique():
      subEntries = entries.loc[entries['courseTaught'] == subject]
      rowdict = {'Instructor' : instructor}
      rowdict['Subject'] = subject
      if weighting is not None:
        rowdict['avNormBenefit'] = np.average(subEntries['normBenefit'].apply(float), weights=subEntries[weighting].apply(float))
        rowdict['avGradeBenefit'] = np.average(subEntries['gradeBenefit'].apply(float), weights=subEntries[weighting].apply(float))
        if extraNorm is not None:
          rowdict['av' + extraNorm] = np.average(subEntries[extraNorm].apply(float), weights=subEntries[weighting].apply(float))
      else:
        rowdict['avNormBenefit'] = np.average(subEntries['normBenefit'].apply(float))
        rowdict['avGradeBenefit'] = np.average(subEntries['gradeBenefit'].apply(float))
        if extraNorm is not None:
          rowdict['av' + extraNorm] = np.average(subEntries[extraNorm].apply(float))
      rowdict['students/entries'] = sum(subEntries['#students'].apply(int))
      rowdict['avStudents'] = np.average(subEntries['#students'].apply(int))
      rowdict['av%ofStudents'] = np.average(subEntries['%ofStudents'].apply(float))
      rowlist.append(rowdict)
  if extraNorm is not None:
    completeDf = pd.DataFrame(rowlist, columns=['Instructor','Subject','avNormBenefit','avGradeBenefit','av' + extraNorm,'students/entries','avStudents', 'av%ofStudents'])
  else:
    completeDf = pd.DataFrame(rowlist, columns=['Instructor','Subject','avNormBenefit','avGradeBenefit','students/entries','avStudents', 'av%ofStudents'])
  completeDf.sort_values(by=['Subject','Instructor'])
  completeDf.reset_index(inplace = True, drop=True)
  completeDf['grade*Norm*Sign(norm)'] = completeDf['avGradeBenefit'] * completeDf['avNormBenefit'] * np.sign(completeDf['avNormBenefit'])
  completeDf['avNormBenefit' + pvalSuffix] = pvalOfSeries(completeDf['avNormBenefit'])
  completeDf['avGradeBenefit' + pvalSuffix] = pvalOfSeries(completeDf['avGradeBenefit'])
  completeDf['grade*Norm*Sign(norm)' + pvalSuffix] = pvalOfSeries(completeDf['grade*Norm*Sign(norm)'])
  if extraNorm is not None:
    completeDf['av' + extraNorm + pvalSuffix] = pvalOfSeries(completeDf['av' + extraNorm])

  if not filename.endswith('.csv'):
    filename = "".join((filename, '.csv'))
  completeDf.to_csv(filename, index=False)
  return completeDf

def instructorAveraging2(data, filename = 'instructorAverages', minPercentage = 5.0, maxPercentage = 90.0, weighting = '#students', extraNorm = None):
  cols = ['Instructor','courseTaught','futureCourse','normBenefit','gradeBenefit','#students', '#nonStudents', 'totalStudents', '%ofStudents']
  if not all(x in data.columns for x in cols):
    print('Error: Columns missing. Did you use the instructorRanksAllClasses method?')
    return
  if weighting not in data.columns and weighting is not None:
    print('Error: Weight not present in columns. Check spelling.')
    return
  filteredData = data.loc[(data['%ofStudents'].apply(float) <= maxPercentage) & (data['%ofStudents'].apply(float) >= minPercentage)]
  filteredData['courseTaught'] = data['courseTaught'].str.replace('\d+', '')
  filteredData['futureCourse'] = data['futureCourse'].str.replace('\d+', '')
  uniqueInstructors = filteredData['Instructor'].unique()
  grouped = filteredData.groupby('Instructor')
  rowlist = []
  for instructor in uniqueInstructors:
    entries = grouped.get_group(instructor)
    for subject in entries['courseTaught'].unique():
      subEntries = entries.loc[entries['courseTaught'] == subject]
      for subject2 in subEntries['futureCourse'].unique():
        sub2Entries = subEntries.loc[subEntries['futureCourse'] == subject2]
        rowdict = {'Instructor' : instructor}
        rowdict['firstSubject'] = subject
        rowdict['secondSubject'] = subject2
        if weighting is not None:
          rowdict['avNormBenefit'] = np.average(sub2Entries['normBenefit'].apply(float), weights=sub2Entries[weighting].apply(float))
          rowdict['avGradeBenefit'] = np.average(sub2Entries['gradeBenefit'].apply(float), weights=sub2Entries[weighting].apply(float))
          if extraNorm is not None:
            rowdict['av' + extraNorm] = np.average(sub2Entries[extraNorm].apply(float), weights=sub2Entries[weighting].apply(float))
        else:
          rowdict['avNormBenefit'] = np.average(sub2Entries['normBenefit'].apply(float))
          rowdict['avGradeBenefit'] = np.average(sub2Entries['gradeBenefit'].apply(float))
          if extraNorm is not None:
            rowdict['av' + extraNorm] = np.average(sub2Entries[extraNorm].apply(float))
        rowdict['students/entries'] = sum(sub2Entries['#students'].apply(int))
        rowdict['avStudents'] = np.average(sub2Entries['#students'].apply(int))
        rowdict['av%ofStudents'] = np.average(sub2Entries['%ofStudents'].apply(float))
      rowlist.append(rowdict)
  if extraNorm is not None:
    completeDf = pd.DataFrame(rowlist, columns=['Instructor','firstSubject','secondSubject','avNormBenefit','avGradeBenefit','av' + extraNorm,'students/entries','avStudents', 'av%ofStudents'])
  else:
    completeDf = pd.DataFrame(rowlist, columns=['Instructor','firstSubject','secondSubject','avNormBenefit','avGradeBenefit','students/entries','avStudents', 'av%ofStudents'])
  completeDf.sort_values(by=['secondSubject','firstSubject','Instructor'])
  completeDf.reset_index(inplace = True, drop=True)
  completeDf['grade*Norm*Sign(norm)'] = completeDf['avGradeBenefit'] * completeDf['avNormBenefit'] * np.sign(completeDf['avNormBenefit'])
  completeDf['avNormBenefit' + pvalSuffix] = pvalOfSeries(completeDf['avNormBenefit'])
  completeDf['avGradeBenefit' + pvalSuffix] = pvalOfSeries(completeDf['avGradeBenefit'])
  completeDf['grade*Norm*Sign(norm)' + pvalSuffix] = pvalOfSeries(completeDf['grade*Norm*Sign(norm)'])
  if extraNorm is not None:
    completeDf['av' + extraNorm + pvalSuffix] = pvalOfSeries(completeDf['av' + extraNorm])

  if not filename.endswith('.csv'):
    filename = "".join((filename, '.csv'))
  completeDf.to_csv(filename, index=False)
  return completeDf

def analyzePairs(data):
  data[data.columns[1]] = data[data.columns[1]].str.replace('\d+', '')
  data[data.columns[2]] = data[data.columns[2]].str.replace('\d+', '')
  same = sum(data[data.columns[1]] == data[data.columns[2]])
  # different = len(data.index) - same
  res = same / len(data.index)
  print(str(same) + ' / ' + str(len(data.index)) + ' or ' + str(round(res*100,3)) + '%')
  return res

def pvalOfSeries(data):
  # data.astype(float, copy=False)
  data = data.apply(float)
  normMean, normStd = data.mean(), data.std()
  # zval = lambda x: (x - normMean) / normStd
  # pval = lambda x: (1 - sciNorm.cdf(abs(x))) * 2
  # return data.apply(zval).apply(pval)
  # print('converted, did mean and std')
  # zval = lambda x: (x - normMean) / normStd
  pval = lambda x: (1 - sciNorm.cdf(abs((x - normMean) / normStd))) * 2
  return pval(data.values)

def tTestOfTwoSeries(data1,data2):
  data1,data2 = data1.apply(float),data2.apply(float)
  data1m, data1std = data1.mean(), data1.std()
  data2m, data2std = data2.mean(), data2.std()
  data1n, data2n = data1.size, data2.size
  tTest = (data1m - data2m) / math.sqrt(((data1std**2)/data1n) + ((data2std**2)/data2n))
  return tTest

def seriesToHistogram(data, fileName = 'histogram', graphTitle='Distribution', sortedAscending = True, logScale = False,xlbl='Value',ylbl = 'Frequency'):
  data2 = data.replace(' ', np.nan)
  data2.dropna(inplace=True)
  # data2.sort_values(inplace=True)
  try:
    histData = pd.to_numeric(data2, errors='raise')
    numericData = True
  except:
    histData = data2
    numericData = False
  if numericData:
    # frequencies, edges = np.histogram(gpas, int((highest - lowest) / 0.1), (lowest, highest))
    
    dataList = histData.tolist()
    frequencies, edges = np.histogram(dataList, (int(math.sqrt(len(dataList))) if (len(dataList) > 30) else (max(len(dataList) // 3 , 1))), (min(dataList), max(dataList)))
    #print('Values: %s, Edges: %s' % (frequencies.shape[0], edges.shape[0]))
    
    if logScale:
      frequencies = [math.log10(freq) if freq > 0 else freq for freq in frequencies]
      ylbl += ' (log 10 scale)'
    histo = hv.Histogram((edges, frequencies))
    histo.opts(opts.Histogram(xlabel=xlbl, ylabel=ylbl, title=graphTitle, fontsize={'title': 40, 'labels': 20, 'xticks': 20, 'yticks': 20}))
    subtitle= 'mean: ' + str(round(sum(dataList) / len(dataList), 3))+ ', n = ' + str(len(dataList))
    hv.output(size=250)
    graph = hv.render(histo)
    graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="30pt"), 'above')
    output_file(outDir +fileName + '.html', mode='inline')
    save(graph)
    show(graph)
    if not edmApplication:
      hv.output(size=300)
      histo.opts(toolbar=None)
      graph = hv.render(histo)
      graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="30pt"), 'above')
      export_png(graph, filename=outDir +fileName + '.png')
  else:
    barData = histData.value_counts(dropna=False)
    dictList = sorted(zip(barData.index, barData.values), key = lambda x: x[sortedAscending])
    # print(dictList)
    bar = hv.Bars(dictList)
    bar.opts(opts.Bars(xlabel=xlbl, ylabel=ylbl, title=graphTitle))
    subtitle= 'n = ' + str(len(dictList))
    hv.output(size=250)
    graph = hv.render(bar)
    graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="30pt"), 'above')
    output_file(outDir +fileName + '.html', mode='inline')
    save(graph)
    show(graph)
    if not edmApplication:
      hv.output(size=300)
      bar.opts(toolbar=None)
      graph2 = hv.render(bar)
      graph2.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="30pt"), 'above')
      export_png(graph2, filename=outDir +fileName + '.png')
  hv.output(size=125)
    
