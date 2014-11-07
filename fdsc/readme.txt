Fault Displacement Slip-Curve (FDSC) v1.0

Copyright (C) (2013-2014) Ana-Maria Rosu
IPGP-ENSG/IGN project financed by TOSCA/CNES

This software is a computer program whose purpose is to outline the
track of a fault, to do perpendicular profiles to this track, stack them
and have the slip-curve plot as an output.

This software is governed by the CeCILL-B license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL-B
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL-B license and that you accept its terms.


////////////////////////////////////////////////////////////////////////

Dependencies
  python2.7
  gdal-bin
  python-gdal
  python-matplotlib
  python-qt4
  python-scipy

Usage
  Command:
    python fdsc.py

  Drawing the fault trace:
    draw: mouse left-click
    erase the last point: mouse middle-click
    save fault trace: mouse right-click

  Offsets tool:
    there are 4 points (<=> 2 line segments) 'framing' the offset
    to change a point's position: left-click on it and another left-click on the new position of the point
    the offset is computed between the 3rd and the 2nd point
    to save the offset: right-click
    all the offset values are saved into a file, as well as other information regarding the stacking parameters

  Slip-curve tool:
    plotting the offsets along the fault following column, line, parallel or perpendicular directions
      abscissa: distance along the fault
      ordinate: offset values



X,Y in image frame
X: columns
Y: lines
____________ X
|
|
|
Y

////////////////////////////////////////////////////////////////////////

