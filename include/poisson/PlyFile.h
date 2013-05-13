/*

 Header for PLY polygon files.
 
  - Greg Turk, March 1994
  
   A PLY file contains a single polygonal _object_.
   
	An object is composed of lists of _elements_.  Typical elements are
	vertices, faces, edges and materials.
	
	 Each type of element for a given object has one or more _properties_
	 associated with the element type.  For instance, a vertex element may
	 have as properties three floating-point values x,y,z and three unsigned
	 chars for red, green and blue.
	 
	  ---------------------------------------------------------------
	  
	   Copyright (c) 1994 The Board of Trustees of The Leland Stanford
	   Junior University.  All rights reserved.   
	   
		Permission to use, copy, modify and distribute this software and its   
		documentation for any purpose is hereby granted without fee, provided   
		that the above copyright notice and this permission notice appear in   
		all copies of this software and that you do not sell the software.   
		
		 THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,   
		 EXPRESS, IMPLIED OR OTHERWISE, INCLUDING WITHOUT LIMITATION, ANY   
		 WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.   
		 
*/

#ifndef PLY_FILE_INCLUDED
#define PLY_FILE_INCLUDED


#ifndef WIN32
	#ifdef __cplusplus
	extern "C" {
	#endif
#endif //WIN32

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <string>
    
#define PLY_ASCII         1      /* ascii PLY file */
#define PLY_BINARY_BE     2      /* binary PLY file, big endian */
#define PLY_BINARY_LE     3      /* binary PLY file, little endian */
#define PLY_BINARY_NATIVE 4      /* binary PLY file, same endianness as current architecture */
    
#define PLY_OKAY    0           /* ply routine worked okay */
#define PLY_ERROR  -1           /* error in ply routine */
	
	/* scalar data types supported by PLY format */
	
#define PLY_START_TYPE 0
#define PLY_CHAR       1
#define PLY_SHORT      2
#define PLY_INT        3
#define PLY_UCHAR      4
#define PLY_USHORT     5
#define PLY_UINT       6
#define PLY_FLOAT      7
#define PLY_DOUBLE     8
#define PLY_INT_8      9
#define PLY_UINT_8     10
#define PLY_INT_16     11
#define PLY_UINT_16    12
#define PLY_INT_32     13
#define PLY_UINT_32    14
#define PLY_FLOAT_32   15
#define PLY_FLOAT_64   16
	
#define PLY_END_TYPE   17
	
#define  PLY_SCALAR  0
#define  PLY_LIST    1
	
#define PLY_STRIP_COMMENT_HEADER 0

/*** delcaration of routines ***/

extern PlyFile *ply_write(FILE *, int, string *, int);
extern PlyFile *ply_open_for_writing(char *, int, string *, int, float *);
extern void ply_describe_element(PlyFile *, char *, int, int, PlyProperty *);
extern void ply_describe_property(PlyFile *, const string &, PlyProperty *);
extern void ply_element_count(PlyFile *, const string &, int);
extern void ply_header_complete(PlyFile *);
extern void ply_put_element_setup(PlyFile *, const string &);
extern void ply_put_element(PlyFile *, void *);
extern void ply_put_comment(PlyFile *, char *);
extern void ply_put_obj_info(PlyFile *, char *);
extern PlyFile *ply_read(FILE *, int *, char ***);
extern PlyFile *ply_open_for_reading( char *, int *, char ***, int *, float *);
extern PlyProperty **ply_get_element_description(PlyFile *, char *, int*, int*);
extern void ply_get_element_setup( PlyFile *, char *, int, PlyProperty *);
extern int ply_get_property(PlyFile *, char *, PlyProperty *);
extern PlyOtherProp *ply_get_other_properties(PlyFile *, char *, int);
extern void ply_get_element(PlyFile *, void *);
extern char **ply_get_comments(PlyFile *, int *);
extern char **ply_get_obj_info(PlyFile *, int *);
extern void ply_close(PlyFile *);
extern void ply_get_info(PlyFile *, float *, int *);
extern PlyOtherElems *ply_get_other_element (PlyFile *, char *, int);
extern void ply_describe_other_elements ( PlyFile *, PlyOtherElems *);
extern void ply_put_other_elements (PlyFile *);
extern void ply_free_other_elements (PlyOtherElems *);
extern void ply_describe_other_properties(PlyFile *, PlyOtherProp *, int);

extern int equal_strings(const char *, const char *);

#ifndef WIN32
	#ifdef __cplusplus
	}
	#endif
#endif

#endif // PLY_FILE_INCLUDED
