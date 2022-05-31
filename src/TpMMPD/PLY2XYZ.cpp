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

#include "StdAfx.h"

#if ELISE_QT 

	#include "../saisieQT/include_QT/Cloud.h"
	
	int PLY2XYZ_main(int argc,char ** argv)
	{
		std::string aPly, aOut, aDir;
		ElInitArgMain
		(
			argc,argv,
			LArgMain()  << EAMC(aDir,"Directory")
						<< EAMC(aPly,"Ply file to export", eSAM_IsExistFile),
			LArgMain()  << EAM(aOut,"Out",false,"Name output file (def=plyNameFile.xyz)")
							
		);
				
		if (!EAMIsInit(&aOut)) aOut = StdPrefix(aPly) + ".xyz";
		

		
		vector <GlCloud*> clouds;
		GlCloud * cloud = GlCloud::loadPly(aDir + ELISE_CAR_DIR + aPly);
		if (cloud)
		{
			clouds.push_back( cloud );
		}
			
		//Data
		if (!MMVisualMode)
		{
			FILE * aFP = FopenNN(aOut,"w","PLY2XYZ_main");
				
			cElemAppliSetFile aEASF(aDir + ELISE_CAR_DIR + aOut);
				
			for (int aK=0 ; aK< (int) clouds.size() ; aK++)
			{
				GlCloud *cloud = clouds[aK];

				for (int bK=0; bK < (int) cloud->size(); ++bK)
				{
					GlVertex vertex = cloud->getVertex(bK);
					QVector3D pt = vertex.getPosition();
					fprintf(aFP,"%lf %lf %lf \n",pt.x(),pt.y(),pt.z());
				}
			}
			
		ElFclose(aFP);
			
		}
		

        return EXIT_SUCCESS;
	}
		
		
	
#else
	int PLY2XYZ_main(int argc,char ** argv)
	{
		std::string aPly, aOut, aDir;
		ElInitArgMain
		(
			argc,argv,
			LArgMain()  << EAMC(aDir,"Directory")
						<< EAMC(aPly,"Ply file to export", eSAM_IsExistFile),
			LArgMain()  << EAM(aOut,"Out",false,"Name output file (def=plyNameFile.xyz)")
							
		);
				
		if (!EAMIsInit(&aOut)) aOut = StdPrefix(aPly) + ".xyz";
	
		list<string> aVFiles = RegexListFileMatch(aDir, aPly, 1, false);

        sPlyOrientedColoredAlphaVertex **glist=NULL;
        int gen_nelems =0;
        int Cptr = 0;

        int type = 0;
	if (type) {} // Warning setbutnotused

        bool wNormales = false;

        PlyFile * thePlyFile;
        int nelems, nprops, num_elems, file_type;
        float version;
        char **elist;
        char *elem_name;
        PlyProperty **plist=NULL;

        //get global number of elements
        list<string>::iterator itr = aVFiles.begin();
        for(;itr != aVFiles.end(); itr++)
        {
            thePlyFile = ply_open_for_reading( const_cast<char *>((aDir + ELISE_CAR_DIR + (*itr)).c_str()), &nelems, &elist, &file_type, &version);

            cout << "loading file " << *itr	<< endl;
#ifdef _DEBUG
            cout << "version "	<< version		<< endl;
            cout << "type "		<< file_type	<< endl;
            cout << "nb elem "	<< nelems		<< endl;
#endif

            elem_name = elist[0];
            plist = ply_get_element_description (thePlyFile, elem_name, &num_elems, &nprops);

            gen_nelems += num_elems;

            ply_close (thePlyFile);
        }

        cout << "nb total elem "	<< gen_nelems << endl;
        glist = (sPlyOrientedColoredAlphaVertex **) malloc (sizeof (sPlyOrientedColoredAlphaVertex *) * gen_nelems);

        //read ply files
        itr = aVFiles.begin();
        for(;itr != aVFiles.end(); itr++)
        {
            thePlyFile = ply_open_for_reading( const_cast<char *>((aDir + ELISE_CAR_DIR +(*itr)).c_str()), &nelems, &elist, &file_type, &version);

            for (int i = 0; i < nelems; i++)
            {
                // get the description of the first element
                elem_name = elist[i];
                plist = ply_get_element_description (thePlyFile, elem_name, &num_elems, &nprops);

                if (equal_strings ("vertex", elem_name))
                {
                    printf ("element %s number= %d\n", elem_name, num_elems);

                    switch(nprops)
                    {
                        case 10: // x y z nx ny nz r g b a
                        {
                            type = 5;
                            for (int j = 0; j < nprops ;++j)
                                ply_get_property (thePlyFile, elem_name, &oriented_colored_alpha_vert_props[j]);

                            sPlyOrientedColoredAlphaVertex *vertex = (sPlyOrientedColoredAlphaVertex *) malloc (sizeof (sPlyOrientedColoredAlphaVertex));

                            // grab all the vertex elements
                            for (int j = 0; j < num_elems; j++, Cptr++)
                            {
                                ply_get_element (thePlyFile, (void *) vertex);

        #ifdef _DEBUG
            printf ("vertex--: %g %g %g %g %g %g %u %u %u %u\n", vertex->x, vertex->y, vertex->z, vertex->nx, vertex->ny, vertex->nz, vertex->red, vertex->green, vertex->blue, vertex->alpha);
        #endif

                                glist[Cptr] = vertex;
                            }
                            break;
                        }
                        case 9: // x y z nx ny nz r g b
                        {
                            type = 4;
                            for (int j = 0; j < nprops ;++j)
                                ply_get_property (thePlyFile, elem_name, &oriented_colored_vert_props[j]);

                            sPlyOrientedColoredVertex *vertex = (sPlyOrientedColoredVertex *) malloc (sizeof (sPlyOrientedColoredVertex));

                            // grab all the vertex elements
                            for (int j = 0; j < num_elems; j++, Cptr++)
                            {

                                ply_get_element (thePlyFile, (void *) vertex);

        #ifdef _DEBUG
            printf ("vertex--: %g %g %g %g %g %g %u %u %u\n", vertex->x, vertex->y, vertex->z, vertex->nx, vertex->ny, vertex->nz, vertex->red, vertex->green, vertex->blue);
        #endif

                                sPlyOrientedColoredAlphaVertex *fvertex = (sPlyOrientedColoredAlphaVertex *) malloc (sizeof (sPlyOrientedColoredAlphaVertex));

                                fvertex->x = vertex->x;
                                fvertex->y = vertex->y;
                                fvertex->z = vertex->z;

                                fvertex->nx = vertex->nx;
                                fvertex->ny = vertex->ny;
                                fvertex->nz = vertex->nz;

                                fvertex->red   = vertex->red;
                                fvertex->green = vertex->green;
                                fvertex->blue  = vertex->blue;

                                glist[Cptr] = fvertex;

                            }
                            break;
                        }
                        case 7:
                        {
                            type = 2;
                            // setup for getting vertex elements
                            for (int j = 0; j < nprops ;++j)
                                ply_get_property (thePlyFile, elem_name, &colored_a_vert_props[j]);

                            sPlyColoredVertexWithAlpha * vertex = (sPlyColoredVertexWithAlpha *) malloc (sizeof (sPlyColoredVertexWithAlpha));

                            // grab all the vertex elements
                            for (int j = 0; j < num_elems; j++, Cptr++)
                            {
                                ply_get_element (thePlyFile, (void *) vertex);

                                #ifdef _DEBUG
                                    printf ("vertex--: %g %g %g %u %u %u %u\n", vertex->x, vertex->y, vertex->z, vertex->red, vertex->green, vertex->blue, vertex->alpha);
                                #endif

                                sPlyOrientedColoredAlphaVertex *fvertex = (sPlyOrientedColoredAlphaVertex *) malloc (sizeof (sPlyOrientedColoredAlphaVertex));

                                fvertex->x = vertex->x;
                                fvertex->y = vertex->y;
                                fvertex->z = vertex->z;

                                fvertex->red   = vertex->red;
                                fvertex->green = vertex->green;
                                fvertex->blue  = vertex->blue;
                                fvertex->alpha = vertex->alpha;

                                glist[Cptr] = fvertex;
                            }
                            break;
                        }
                        case 6:
                        {
                            // can be (x y z r g b) or (x y z nx ny nz)
                            PlyElement *elem = NULL;

                            for (int i = 0; i < nelems; i++)
                                if (equal_strings ("vertex", thePlyFile->elems[i]->name))
                                    elem = thePlyFile->elems[i];

                            for (int i = 0; i < nprops; i++)
                                if ( "nx"==elem->props[i]->name )   wNormales = true;

                            if (!wNormales)
                            {
                                type = 1;
                                for (int j = 0; j < nprops ;++j)
                                    ply_get_property (thePlyFile, elem_name, &colored_vert_props[j]);

                                sPlyColoredVertex *vertex = (sPlyColoredVertex *) malloc (sizeof (sPlyColoredVertex));

                                for (int j = 0; j < num_elems; j++, Cptr++)
                                {

                                    ply_get_element (thePlyFile, (void *) vertex);

                                    #ifdef _DEBUG
                                        printf ("vertex: %g %g %g %u %u %u\n", vertex->x, vertex->y, vertex->z, vertex->red, vertex->green, vertex->blue);
                                    #endif

                                        sPlyOrientedColoredAlphaVertex *fvertex = (sPlyOrientedColoredAlphaVertex *) malloc (sizeof (sPlyOrientedColoredAlphaVertex));

                                        fvertex->x = vertex->x;
                                        fvertex->y = vertex->y;
                                        fvertex->z = vertex->z;

                                        fvertex->red   = vertex->red;
                                        fvertex->green = vertex->green;
                                        fvertex->blue  = vertex->blue;

                                        glist[Cptr] = fvertex;
                                }
                            }
                            else
                            {
                                type = 3;
                                for (int j = 0; j < nprops ;++j)
                                    ply_get_property (thePlyFile, elem_name, &oriented_vert_props[j]);

                                sPlyOrientedVertex *vertex = (sPlyOrientedVertex *) malloc (sizeof (sPlyOrientedVertex));

                                for (int j = 0; j < num_elems; j++, Cptr++)
                                {
                                    ply_get_element (thePlyFile, (void *) vertex);

                                    #ifdef _DEBUG
                                        printf ("vertex: %g %g %g %g %g %g\n", vertex->x, vertex->y, vertex->z, vertex->nx, vertex->ny, vertex->nz);
                                    #endif

                                    sPlyOrientedColoredAlphaVertex *fvertex = (sPlyOrientedColoredAlphaVertex *) malloc (sizeof (sPlyOrientedColoredAlphaVertex));

                                    fvertex->x = vertex->x;
                                    fvertex->y = vertex->y;
                                    fvertex->z = vertex->z;

                                    fvertex->nx = vertex->nx;
                                    fvertex->ny = vertex->ny;
                                    fvertex->nz = vertex->nz;

                                    glist[Cptr] = fvertex;
                                }
                            }
                            break;
                        }
                        case 3:
                        {
                            for (int j = 0; j < nprops ;++j)
                                ply_get_property (thePlyFile, elem_name, &vert_props[j]);

                            sVertex *vertex = (sVertex *) malloc (sizeof (sVertex));

                            for (int j = 0; j < num_elems; j++, Cptr++)
                            {

                                ply_get_element (thePlyFile, (void *) vertex);

            #ifdef _DEBUG
                                printf ("vertex: %g %g %g\n", vertex->x, vertex->y, vertex->z);
            #endif

                                sPlyOrientedColoredAlphaVertex *fvertex = (sPlyOrientedColoredAlphaVertex *) malloc (sizeof (sPlyOrientedColoredAlphaVertex));

                                fvertex->x = vertex->x;
                                fvertex->y = vertex->y;
                                fvertex->z = vertex->z;

                                glist[Cptr] = fvertex;
                            }
                            break;
                        }
                        default:
                        {
                            printf("unable to load a ply unless number of properties is 3, 6, 7, 9 or 10\n");
                            break;
                        }
                    }
                }
            }

            ply_close (thePlyFile);
        }
			
		//write ply file in text
		if (!MMVisualMode)
		{
			FILE * aFP = FopenNN(aOut,"w","PLY2XYZ_main");
			cElemAppliSetFile aEASF(aDir + ELISE_CAR_DIR + aOut);
			for (int aK=0 ; aK< gen_nelems ; aK++)
			{
				sPlyOrientedColoredAlphaVertex * pt = glist[aK];
				fprintf(aFP,"%.7f %.7f %.7f\n", pt->x, pt->y, pt->z);       
			}
		ElFclose(aFP);
		}
			
		if ( glist!=NULL ) free(glist); // G++11 delete glist;
		if ( plist!=NULL ) delete plist;
			
			
		return EXIT_SUCCESS;
}
#endif
/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant Ã  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est rÃ©gi par la licence CeCILL-B soumise au droit franÃ§ais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusÃ©e par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilitÃ© au code source et des droits de copie,
de modification et de redistribution accordÃ©s par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitÃ©e.  Pour les mÃªmes raisons,
seule une responsabilitÃ© restreinte pÃ¨se sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concÃ©dants successifs.

A cet Ã©gard  l'attention de l'utilisateur est attirÃ©e sur les risques
associÃ©s au chargement,  Ã  l'utilisation,  Ã  la modification et/ou au
dÃ©veloppement et Ã  la reproduction du logiciel par l'utilisateur Ã©tant
donnÃ© sa spÃ©cificitÃ© de logiciel libre, qui peut le rendre complexe Ã
manipuler et qui le rÃ©serve donc Ã  des dÃ©veloppeurs et des professionnels
avertis possÃ©dant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invitÃ©s Ã  charger  et  tester  l'adÃ©quation  du
logiciel Ã  leurs besoins dans des conditions permettant d'assurer la
sÃ©curitÃ© de leurs systÃ¨mes et ou de leurs donnÃ©es et, plus gÃ©nÃ©ralement,
Ã  l'utiliser et l'exploiter dans les mÃªmes conditions de sÃ©curitÃ©.

Le fait que vous puissiez accÃ©der Ã  cet en-tÃªte signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez acceptÃ© les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
