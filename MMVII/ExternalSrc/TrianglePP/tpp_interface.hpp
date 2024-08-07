/**
   @file  tpp_interface.hpp

   @brief Declaration of the main Delaunay class and Iterators of the Triangle++ wrapper

	 This is a wrapper for the original Triangle (aka TriLib) code by J.R. Shewchuk. It can create
	 standard Delaunay triangulations, quality Delaunay triangulations, constrained Delaunay
	 triangulations and, last but not least, also Voronoi diagrams.

	 Original Triangle/TriLib code: http://www.cs.cmu.edu/~quake/triangle.html

   @author  Marek Krajewski (mrkkrj), www.ib-krajewski.de
   @author  Piyush Kumar (piyush), http://compgeom.com/~piyush
   @author  Jonathan Richard Shewchuk (original TriLib!!!), https://people.eecs.berkeley.edu/~jrs/

   @copyright  Copyright 20218, Marek Krajewski, released under the terms of LGPL v3

   @changes
	  11/03/06: piyush - Fixed the compilation system.
	  10/25/06: piyush - Wrapped in tpp namespace for usage with other libraries with similar names.
						 Added some more documentation/small changes. Used doxygen 1.5.0 and dot. Tested
						 compilation with icc 9.0/9.1, gcc-4.1/3.4.6.
	  10/21/06: piyush - Replaced vertexsort with C++ sort.
	  08/24/11: mrkkrj - Ported to Visual Studio, added comp. operators, reformatted and added some comments
	  10/15/11: mrkkrj - added support for the "quality triangulation" option, added some debug support
	  11/07/11: mrkkrj - bugfix in Triangle's divandconqdelauney()
	  17/09/18: mrkkrj � ported to 64-bit (preliminary, not thorougly tested!)
	  22/01/20: mrkkrj � added support for custom constraints (angle and area)
	  17/04/20: mrkkrj � added support for Voronoi tesselation
	  05/08/22: mrkkrj � added more tests for constrained PSLG triangulations, included (reworked) Yejneshwar's
						 fix for removal of concavities
	  17/12/22: mrkkrj � Ported to Linux, reworked Yejneshwar's fix again
	  30/12/22: mrkkrj � added first file read-write support
	  03/02/23: mrkkrj � added first support for input sanitization
	  15/03/23: mrkkrj � added support for iteration over the resulting mesh, some refactorings
	  27/03/23: mrkkrj � API break, removed old (i.e. deprecated) names, changed comment formatting
	  05/07/23: mrkkrj � set TriLib's SELF_CHECK option as default, make it overridable with TRIANGLE_NO_TRILIB_SELFCHECK,
						 bugfix TriLib internal error in deletevertex()
	  29/09/23: mrkkrj � first support for regions and regional constraints
	  24/10/23: mrkkrj � support for DLL builds
*/

#ifndef TRPP_INTERFACE
#define TRPP_INTERFACE

#ifndef TRPP_BUILD_SHARED
#define TRPP_LIB_EXPORT
#else
// DLL build support
#include <tpp_export.h>
#endif

// the triangulator
#include "tpp_delaunay.hpp"

// lazy acces to the results
#include "tpp_iterators.hpp"

// walking around the resulting mesh
#include "tpp_triangulation_mesh.hpp"

#endif
