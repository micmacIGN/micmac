/*eLiSe06/05/99
  
     Copyright (C) 1999 Marc PIERROT DESEILLIGNY

   eLiSe : Elements of a Linux Image Software Environment

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Author: Marc PIERROT DESEILLIGNY    IGN/MATIS  
Internet: Marc.Pierrot-Deseilligny@ign.fr
   Phone: (33) 01 43 98 81 28
eLiSe06/05/99*/



/*

Copyright (C) 1998 Marc PIERROT DESEILLIGNY

   Skeletonization by veinerization. 

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Author: Marc PIERROT DESEILLIGNY    IGN/MATIS
Internet: Marc.Pierrot-Deseilligny@ign.fr
   Phone: (33) 01 43 98 81 28


   Detail of the algoprithm in Deseilligny-Stamon-Suen
   "Veinerization : a New Shape Descriptor for Flexible
    Skeletonization" in IEEE-PAMI Vol 20 Number 5, pp 505-521

    It also give the signification of main parameters.
*/


// A very simple ( and very dirty) example of use.
//  Compile with g++ 2.91.66. No test  on any other compiler

#if (0)

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <string>
#include <new>
#include <cassert>
#include <cfloat>
#include <sys/types.h>
#include <sys/stat.h>
#include <cassert>


#include "export_skel_vein.h"



unsigned char ** image(int tx,int ty)
{
	unsigned char ** res = new unsigned char * [ty];
	for (int y=0; y<ty; y++)
	{
		res[y] = new unsigned char [tx];
		for (int x =0; x< tx; x++)
			res[y][x] =0;
	}

	return res;
}


int main(int,char **)
{
     //++++++++++++++++++++++++++++++++++++++++++
     // Image creation
     //++++++++++++++++++++++++++++++++++++++++++

	int tx = 108;
	int ty = 58;

    unsigned char ** Iin = image(tx,ty);
    unsigned char ** Iout = image(tx,ty);



     //++++++++++++++++++++++++++++++++++++++++++
	 // Initialize Iin wiht file "eLiSe.pgm"
     //++++++++++++++++++++++++++++++++++++++++++

	{
		FILE * fp = fopen("eLiSe.pgm","r");
		assert(fp!=0);
    	fseek(fp,73,SEEK_SET);
		for (int y=0; y< ty; y++)
			fread(Iin[y],sizeof(char),tx,fp);
    	fclose(fp);
	}


     //++++++++++++++++++++++++++++++++++++++++++
	 // Call to skeleton
     //++++++++++++++++++++++++++++++++++++++++++

	VeinerizationSkeleton
	(
		Iout,
		Iin,		
		tx,
		ty,
		8,		// Surfacic threshold
		3.14,   // angular threshold
		true,   // we want disc-like shape to have a skeleton
		true,   // we want skeleton to end until extremities
		false,  // we do not use the results
		(unsigned short ** )0  // we do not give temporary 
	);


     //++++++++++++++++++++++++++++++++++++++++++
	 // Write Iout in SkeLiSe.pgm
     //++++++++++++++++++++++++++++++++++++++++++

	{
		FILE * fp = fopen("SkeLiSe.pgm","w");
		assert(fp!=0);
		fprintf(fp,"P5\n");
		fprintf(fp,"#creator eLiSe \n");
		fprintf(fp,"108 58\n");
		fprintf(fp,"255\n");
		for (int y=0; y< ty; y++)
			fwrite(Iout[y],sizeof(char),tx,fp);
    	fclose(fp);
	}


     //++++++++++++++++++++++++++++++++++++++++++

    
	return 0;
}

#endif






