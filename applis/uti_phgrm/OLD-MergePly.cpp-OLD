/*Header-MicMac-eLiSe-25/06/2007
 
 MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
 eLiSe  : ELements of an Image Software Environnement
 
 www.micmac.ign.fr
 
 
 Copyright : Institut Geographique National
 Author : Marc Pierrot Deseilligny
 Contributors : Gregoire Maillet, Didier Boldo.
 
 [1] M. Pierrot-Deseilligny, N. Paparoditis.
 "A multiresolution and optimization-based image matching approach:
 An application to surface reconstruction from SPOT5-HRS stereo imagery."
 In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
 (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.
 
 [2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
 d'images, adapte au contexte geograhique" to appears in 
 Bulletin d'information de l'Institut Geographique National, 2007.
 
 Francais :
 
 MicMac est un logiciel de mise en correspondance d'image adapte 
 au contexte de recherche en information geographique. Il s'appuie sur
 la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
 licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.
 
 
 English :
 
 MicMac is an open source software specialized in image matching
 for research in geographic information. MicMac is built on the
 eLiSe image library. MicMac is governed by the  "Cecill-B licence".
 See below and http://www.cecill.info.
 
 Header-MicMac-eLiSe-25/06/2007*/

#include "general/all.h"
#include "private/all.h"
#include <algorithm>
#include "private/ply.h"

/* information needed to describe the user's data to the PLY routines */

const char *elem_names[] = { /* list of the kinds of elements in the user's object */
	"vertex", "face"
};

PlyProperty vert_props[] = { /* list of property information for a vertex */
	{"x", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,x), 0, 0, 0, 0},
	{"y", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,y), 0, 0, 0, 0},
	{"z", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,z), 0, 0, 0, 0},
};

static PlyProperty oriented_vert_props[] = {
	{"x",  PLY_FLOAT, PLY_FLOAT, offsetof(PlyOrientedVertex,x ), 0, 0, 0, 0},
	{"y",  PLY_FLOAT, PLY_FLOAT, offsetof(PlyOrientedVertex,y ), 0, 0, 0, 0},
	{"z",  PLY_FLOAT, PLY_FLOAT, offsetof(PlyOrientedVertex,z ), 0, 0, 0, 0},
	{"nx", PLY_FLOAT, PLY_FLOAT, offsetof(PlyOrientedVertex,nx), 0, 0, 0, 0},
	{"ny", PLY_FLOAT, PLY_FLOAT, offsetof(PlyOrientedVertex,ny), 0, 0, 0, 0},
	{"nz", PLY_FLOAT, PLY_FLOAT, offsetof(PlyOrientedVertex,nz), 0, 0, 0, 0}
};

int main(int argc,char ** argv)
{
    std::string aNameFiles,aNameOut;
    std::vector<string> aVFiles, aVCom;
    
	int aBin  = 1;
	int DoNrm = 1;
	
    ElInitArgMain
    (
	 argc,argv,
	 LArgMain()  << EAM(aNameFiles),
	 LArgMain()		<< EAM(aNameOut,"Out",true)
					<< EAM(aVCom,"Comments",true)
					<< EAM(aBin,"Bin",true,"Generate Binary or Ascii (Def=1, Binary)")
					<< EAM(DoNrm,"Normale",true)
	);	
	
	int pos = aNameFiles.find('#',0);
	while (pos>0)
	{
		aVFiles.push_back(aNameFiles.substr(0,pos));
		aNameFiles = aNameFiles.substr(pos+1, aNameFiles.size());
		pos = aNameFiles.find('#',0);
	}
	aVFiles.push_back(aNameFiles);
	
    if (aNameOut=="")
		aNameOut = StdPrefix(aVFiles[0]) + "_merged.ply";
	
	PlyOrientedVertex **glist;
	int gen_nelems =0;
	int Cptr = 0;
	
	PlyFile * thePlyFile;
	int nelems;
	char **elist;
	int file_type;
	float version;
	int nprops;
	int num_elems;
	char *elem_name;
	PlyProperty **plist;
	PlyOrientedVertex **vlist;
	
	//get global number of elements
	for (unsigned int aK=0; aK< aVFiles.size(); ++aK) 
	{
		thePlyFile = ply_open_for_reading( const_cast<char *>(aVFiles[aK].c_str()), &nelems, &elist, &file_type, &version);
		
		cout << "file "		<< aVFiles[aK]	<< endl;
		cout << "version "	<< version		<< endl;
		cout << "type "		<< file_type	<< endl;
		cout << "nb elem "	<< nelems		<< endl;		
			
		elem_name = elist[0];
		plist = ply_get_element_description (thePlyFile, elem_name, &num_elems, &nprops);
		
		gen_nelems += num_elems;
		
		ply_close (thePlyFile);
	}
	
	cout << "nb total elem "	<< gen_nelems << endl;	
	glist = (PlyOrientedVertex **) malloc (sizeof (PlyOrientedVertex *) * gen_nelems);

	//read ply files
	for (unsigned int aK=0; aK< aVFiles.size(); ++aK) 
	{
		thePlyFile = ply_open_for_reading( const_cast<char *>(aVFiles[aK].c_str()), &nelems, &elist, &file_type, &version);
		
		for (int i = 0; i < nelems; i++) 
		{
			/* get the description of the first element */
			elem_name = elist[i];
			plist = ply_get_element_description (thePlyFile, elem_name, &num_elems, &nprops);
					
			/* print the name of the element, for debugging */
			printf ("element %s %d\n", elem_name, num_elems);
						
			/* if we're on vertex elements, read them in */
			if (equal_strings ("vertex", elem_name)) {
				
				/* create a vertex list to hold all the vertices */
				vlist = (PlyOrientedVertex **) malloc (sizeof (PlyOrientedVertex *) * num_elems);
				
				/* set up for getting vertex elements */
				ply_get_property (thePlyFile, elem_name, &oriented_vert_props[0]);
				ply_get_property (thePlyFile, elem_name, &oriented_vert_props[1]);
				ply_get_property (thePlyFile, elem_name, &oriented_vert_props[2]);
				ply_get_property (thePlyFile, elem_name, &oriented_vert_props[3]);
				ply_get_property (thePlyFile, elem_name, &oriented_vert_props[4]);
				ply_get_property (thePlyFile, elem_name, &oriented_vert_props[5]);
				
				/* grab all the vertex elements */
				for (int j = 0; j < num_elems; j++, Cptr++) 
				{
					/* grab and element from the file */
					vlist[j] = (PlyOrientedVertex *) malloc (sizeof (PlyOrientedVertex));
					
					ply_get_element (thePlyFile, (void *) vlist[j]);
										
					glist[Cptr] = (PlyOrientedVertex *) malloc (sizeof (PlyOrientedVertex)); 
					glist[Cptr] = vlist[j];
					
					/* print out vertex x,y,z for debugging */
					//printf ("vertex: %g %g %g %g %g %g\n", vlist[j]->x, vlist[j]->y, vlist[j]->z, vlist[j]->nx, vlist[j]->ny, vlist[j]->nz);
				}
			}
		}
		
		ply_close (thePlyFile);
	}
    
	//write ply file
	
	//Mode Ecriture : binaire ou non
	std::string mode = aBin ? "wb" : "w";
	FILE * aFP = FopenNN(aNameOut,mode,"MergePly");
	
	//Header
	fprintf(aFP,"ply\n");
	std::string aBinSpec = MSBF_PROCESSOR() ? "binary_big_endian":"binary_little_endian" ;
	
	fprintf(aFP,"format %s 1.0\n",aBin?aBinSpec.c_str():"ascii");
	
	for 
		(
		 std::vector<std::string>::const_iterator itS=aVCom.begin();
		 itS!=aVCom.end();
		 itS++
		 )
	{
		fprintf(aFP,"comment %s\n",itS->c_str());
	}
	fprintf(aFP,"element vertex %d\n", gen_nelems);
	fprintf(aFP,"property float x\n");
	fprintf(aFP,"property float y\n");
	fprintf(aFP,"property float z\n");
	
	if (DoNrm)
	{
		fprintf(aFP,"property float nx\n");
		fprintf(aFP,"property float ny\n");
		fprintf(aFP,"property float nz\n");
	}
	
	fprintf(aFP,"element face %d\n",0);
	fprintf(aFP,"property list uchar int vertex_indices\n");
	fprintf(aFP,"end_header\n");
		
	//data
	for (int aK=0 ; aK< gen_nelems ; aK++)
	{
		//printf ("gen vertex: %g %g %g %g %g %g\n", glist[aK]->x, glist[aK]->y, glist[aK]->z, glist[aK]->nx, glist[aK]->ny, glist[aK]->nz);

		if (aBin)
		{
			WriteType(aFP,float(glist[aK]->x));
			WriteType(aFP,float(glist[aK]->y));
			WriteType(aFP,float(glist[aK]->z));
			
			WriteType(aFP,float(glist[aK]->nx));
			WriteType(aFP,float(glist[aK]->ny));
			WriteType(aFP,float(glist[aK]->nz));
		}
		else
		{
			fprintf(aFP,"%.3f %.3f %.3f %.3f %.3f %.3f\n", glist[aK]->x, glist[aK]->y, glist[aK]->z, glist[aK]->nx, glist[aK]->ny, glist[aK]->nz);
		}
	}
		
	ElFclose(aFP);
	
	delete glist;
}





/*Footer-MicMac-eLiSe-25/06/2007
 
 Ce logiciel est un programme informatique servant à la mise en
 correspondances d'images pour la reconstruction du relief.
 
 Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
 respectant les principes de diffusion des logiciels libres. Vous pouvez
 utiliser, modifier et/ou redistribuer ce programme sous les conditions
 de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
 sur le site "http://www.cecill.info".
 
 En contrepartie de l'accessibilité au code source et des droits de copie,
 de modification et de redistribution accordés par cette licence, il n'est
 offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
 seule une responsabilité restreinte pèse sur l'auteur du programme,  le
 titulaire des droits patrimoniaux et les concédants successifs.
 
 A cet égard  l'attention de l'utilisateur est attirée sur les risques
 associés au chargement,  à l'utilisation,  à la modification et/ou au
 développement et à la reproduction du logiciel par l'utilisateur étant 
 donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
 manipuler et qui le réserve donc à des développeurs et des professionnels
 avertis possédant  des  connaissances  informatiques approfondies.  Les
 utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
 logiciel à leurs besoins dans des conditions permettant d'assurer la
 sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
 à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 
 
 Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
 pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
 termes.
 Footer-MicMac-eLiSe-25/06/2007*/


