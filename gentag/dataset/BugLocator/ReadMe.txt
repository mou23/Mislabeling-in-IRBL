README:

This tool implements an information retrieval based method for locating the relevant files for fixing a bug. It ranks all files based on the textual similarity between the initial bug report and the source code using a revised Vector Space Model (rVSM), taking into consideration information about similar bugs that have been fixed before. 

Requirements:
--------------------------------------------------------------------------
Windows Operating System, JDK 1.6.7+

Tool Usage:
--------------------------------------------------------------------------
java -jar BugLocator.jar [-options] 
where options must include:
-b	indicates the bug information file,the bug information file must be .xml file in the data folder.
-s	indicates the source code directory.
-a	indicates the alpha factor for combining vsmScore and simiScore
-o	indicates the result file,the format of result file is "bug id, relevant source code file,rank(start with 0),score".

for example the command may be "java -jar BugLocator.jar -b E:\Data&Tool\data\SWTBugRepository.xml -s E:\Data&Tool\swt-3.1\src -a 0.2 -o E:\Data&Tool\output.txt" for SWT system



@Copyright Tsinghua University, 2012

- EnD - 