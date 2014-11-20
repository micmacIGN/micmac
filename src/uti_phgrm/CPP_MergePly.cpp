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

void writeHeader(FILE * aFP, int aNelems, int aType, bool aBin)
{
    fprintf(aFP,"ply\n");
    string aBinSpec = MSBF_PROCESSOR() ? "binary_big_endian":"binary_little_endian" ;

    fprintf(aFP,"format %s 1.0\n",aBin?aBinSpec.c_str():"ascii");

    fprintf(aFP,"comment MergePly generated\n");
    fprintf(aFP,"element vertex %d\n", aNelems);

    fprintf(aFP,"property float x\n");
    fprintf(aFP,"property float y\n");
    fprintf(aFP,"property float z\n");

    switch (aType)
    {
        case 0:
            break;
        case 1:
        {
            fprintf(aFP,"property uchar red\n");
            fprintf(aFP,"property uchar green\n");
            fprintf(aFP,"property uchar blue\n");
            break;
        }
        case 2:
        {
            fprintf(aFP,"property uchar red\n");
            fprintf(aFP,"property uchar green\n");
            fprintf(aFP,"property uchar blue\n");
            fprintf(aFP,"property uchar alpha\n");
            break;
        }
        case 3:
        {
            fprintf(aFP,"property float nx\n");
            fprintf(aFP,"property float ny\n");
            fprintf(aFP,"property float nz\n");
            break;
        }
        case 4:
        {
            fprintf(aFP,"property float nx\n");
            fprintf(aFP,"property float ny\n");
            fprintf(aFP,"property float nz\n");
            fprintf(aFP,"property uchar red\n");
            fprintf(aFP,"property uchar green\n");
            fprintf(aFP,"property uchar blue\n");
            break;
        }
        case 5:
        {
            fprintf(aFP,"property float nx\n");
            fprintf(aFP,"property float ny\n");
            fprintf(aFP,"property float nz\n");
            fprintf(aFP,"property uchar red\n");
            fprintf(aFP,"property uchar green\n");
            fprintf(aFP,"property uchar blue\n");
            fprintf(aFP,"property uchar alpha\n");
            break;
        }
    }

    fprintf(aFP,"element face %d\n",0);
    fprintf(aFP,"property list uchar int vertex_indices\n");
    fprintf(aFP,"end_header\n");
}

#if ELISE_QT_VERSION >=4

    #include "../saisieQT/include_QT/Cloud.h"

    int MergePly_main(int argc,char ** argv)
    {
        cout << "using MergePly with QT" << ELISE_QT_VERSION << endl;

        string aFullName,aNameOut;
        string aDir, aPattern;

        bool aBin     = true;

        ElInitArgMain
        (
         argc,argv,
                    LArgMain()	<< EAMC(aFullName, "Full Name (Dir+Pattern)", eSAM_IsPatFile),
                    LArgMain()	<< EAM(aNameOut,"Out",true)
                                << EAM(aBin,"Bin",true,"Generate Binary or Ascii file (Def=true, Binary)", eSAM_IsBool)
        );

        if (MMVisualMode) return EXIT_SUCCESS;

        SplitDirAndFile(aDir, aPattern, aFullName);

        list<string> aVFiles = RegexListFileMatch(aDir, aPattern, 1, false);

        int gen_nelems =0;

        //read ply files
        list<string>::iterator itr = aVFiles.begin();
        int incre=0;
        vector <GlCloud*> clouds;
        for(;itr != aVFiles.end(); itr++, incre++)
        {
            cout << "loading file " << *itr << endl;
            GlCloud * cloud = GlCloud::loadPly(aDir + ELISE_CAR_DIR + *itr, &incre );
            if (cloud)
            {
                clouds.push_back( cloud );
                gen_nelems += cloud->size();
            }
            if (incre>0)
            {
                if (clouds[incre]->type() != clouds[incre-1]->type())
                {
                    cout << "Cant merge ply files from different type (ex: xyz and xyzrgb)" << endl;
                    return EXIT_FAILURE;
                }
            }
        }

        cout << "nb total elem " << gen_nelems << endl;

        int type = clouds[0]->type();

        //write merged file
        if (aNameOut=="")
            aNameOut = aDir + ELISE_CAR_DIR + StdPrefix(*(aVFiles.begin())) + "_merged.ply";


        //Mode Ecriture : binaire ou non
        string mode = aBin ? "wb" : "w";
        FILE * aFP = FopenNN(aNameOut,mode,"MergePly");

        writeHeader(aFP, gen_nelems, type, aBin);

        //Data
        for (int aK=0 ; aK< (int) clouds.size() ; aK++)
        {
            GlCloud *cloud = clouds[aK];

            for (int bK=0; bK < (int) cloud->size(); ++bK)
            {
                GlVertex vertex = cloud->getVertex(bK);
                Pt3dr pt = vertex.getPosition();

                if (aBin)
                {
                    WriteType(aFP,float(pt.x));
                    WriteType(aFP,float(pt.y));
                    WriteType(aFP,float(pt.z));
                }

                switch (type)
                {
                    case 0:
                    {
                        if (!aBin)
                             fprintf(aFP,"%.7f %.7f %.7f\n", pt.x, pt.y, pt.z);
                        break;
                    }
                    case 1:
                    {
                        QColor col = vertex.getColor();
                        if (aBin)
                        {
                            WriteType(aFP,uchar(col.red()));
                            WriteType(aFP,uchar(col.green()));
                            WriteType(aFP,uchar(col.blue()));
                        }
                        else
                            fprintf(aFP,"%.7f %.7f %.7f %d %d %d\n",  pt.x, pt.y, pt.z, col.red(), col.green(), col.blue());

                        break;
                    }
                    case 2:
                    {
                        QColor col = vertex.getColor();

                        if (aBin)
                        {
                            WriteType(aFP,uchar(col.red()));
                            WriteType(aFP,uchar(col.green()));
                            WriteType(aFP,uchar(col.blue()));
                            WriteType(aFP,uchar(col.alpha()));
                        }
                        else
                            fprintf(aFP,"%.7f %.7f %.7f %d %d %d %d\n",  pt.x, pt.y, pt.z, col.red(), col.green(), col.blue(), col.alpha());
                        break;
                    }
                    case 3:
                    {
                        Pt3dr n = vertex.getNormal();

                        if (aBin)
                        {
                            WriteType(aFP,float(n.x));
                            WriteType(aFP,float(n.y));
                            WriteType(aFP,float(n.z));
                        }
                        else
                            fprintf(aFP,"%.7f %.7f %.7f %.7f %.7f %.7f\n",  pt.x, pt.y, pt.z, n.x, n.y, n.z);
                        break;
                    }
                    case 4:
                    {
                        QColor col = vertex.getColor();
                        Pt3dr n = vertex.getNormal();

                        if (aBin)
                        {
                            WriteType(aFP,float(n.x));
                            WriteType(aFP,float(n.y));
                            WriteType(aFP,float(n.z));
                            WriteType(aFP,uchar(col.red()));
                            WriteType(aFP,uchar(col.green()));
                            WriteType(aFP,uchar(col.blue()));
                        }
                        else
                            fprintf(aFP,"%.7f %.7f %.7f %.7f %.7f %.7f %d %d %d\n",  pt.x, pt.y, pt.z, n.x, n.y, n.z, col.red(), col.green(), col.blue() );
                        break;
                    }
                    case 5:
                    {
                        QColor col = vertex.getColor();
                        Pt3dr n = vertex.getNormal();

                        if (aBin)
                        {
                            WriteType(aFP,float(n.x));
                            WriteType(aFP,float(n.y));
                            WriteType(aFP,float(n.z));
                            WriteType(aFP,uchar(col.red()));
                            WriteType(aFP,uchar(col.green()));
                            WriteType(aFP,uchar(col.blue()));
                            WriteType(aFP,uchar(col.alpha()));
                        }
                        else
                            fprintf(aFP,"%.7f %.7f %.7f %.7f %.7f %.7f %d %d %d %d\n",  pt.x, pt.y, pt.z, n.x, n.y, n.z, col.red(), col.green(), col.blue(), col.alpha() );
                        break;
                    }
                }
            }
        }

        ElFclose(aFP);

        return EXIT_SUCCESS;
    }

#else
    #include "../../CodeExterne/Poisson/include/PlyFile.h"

    int MergePly_main(int argc,char ** argv)
    {
        cout << "using MergePly without QT" << endl;

        string aFullName,aNameOut;
        string aDir, aPattern;
        vector<string> aVCom;

        int aBin  = 1;

        ElInitArgMain
        (
         argc,argv,
                    LArgMain()	<< EAMC(aFullName, "Full Name (Dir+Pattern)"),
         LArgMain()		<< EAM(aNameOut,"Out",true)
                        << EAM(aVCom,"Comments",true)
                        << EAM(aBin,"Bin",true,"Generate Binary or Ascii (Def=1, Binary)")
        );

        SplitDirAndFile(aDir, aPattern, aFullName);
        list<string> aVFiles = RegexListFileMatch(aDir, aPattern, 1, false);

        if (aNameOut=="")
            aNameOut = aDir + ELISE_CAR_DIR + StdPrefix(*(aVFiles.begin())) + "_merged.ply";

        sPlyOrientedColoredAlphaVertex **glist=NULL;
        int gen_nelems =0;
        int Cptr = 0;

        int type = 0;
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

        //write ply file

        //Mode Ecriture : binaire ou non
        string mode = aBin ? "wb" : "w";
        FILE * aFP = FopenNN(aNameOut, mode, "MergePly");

        writeHeader(aFP, gen_nelems, type, aBin);

        //data
        for (int aK=0 ; aK< gen_nelems ; aK++)
        {
                sPlyOrientedColoredAlphaVertex * pt = glist[aK];

                if (aBin)
                {
                    WriteType(aFP, pt->x);
                    WriteType(aFP, pt->y);
                    WriteType(aFP, pt->z);
                }

                switch (type)
                {
                    case 0:
                    {
                        if (!aBin)
                             fprintf(aFP,"%.7f %.7f %.7f\n", pt->x, pt->y, pt->z);
                        break;
                    }
                    case 1:
                    {
                        if (aBin)
                        {
                            WriteType(aFP, pt->red );
                            WriteType(aFP, pt->green );
                            WriteType(aFP, pt->blue );
                        }
                        else
                            fprintf(aFP,"%.7f %.7f %.7f %d %d %d\n",  pt->x, pt->y, pt->z, pt->red, pt->green, pt->blue);

                        break;
                    }
                    case 2:
                    {
                        if (aBin)
                        {
                            WriteType(aFP, pt->red);
                            WriteType(aFP, pt->green);
                            WriteType(aFP, pt->blue);
                            WriteType(aFP, pt->alpha);
                        }
                        else
                            fprintf(aFP,"%.7f %.7f %.7f %d %d %d %d\n",  pt->x, pt->y, pt->z, pt->red, pt->green, pt->blue, pt->alpha);
                        break;
                    }
                    case 3:
                    {
                        if (aBin)
                        {
                            WriteType(aFP, pt->nx);
                            WriteType(aFP, pt->ny);
                            WriteType(aFP, pt->nz);
                        }
                        else
                            fprintf(aFP,"%.7f %.7f %.7f %.7f %.7f %.7f\n",  pt->x, pt->y, pt->z, pt->nx, pt->ny, pt->nz);
                        break;
                    }
                    case 4:
                    {
                        if (aBin)
                        {
                            WriteType(aFP, pt->nx);
                            WriteType(aFP, pt->ny);
                            WriteType(aFP, pt->nz);
                            WriteType(aFP, pt->red);
                            WriteType(aFP, pt->green);
                            WriteType(aFP, pt->blue);
                        }
                        else
                            fprintf(aFP,"%.7f %.7f %.7f %.7f %.7f %.7f %d %d %d\n",  pt->x, pt->y, pt->z, pt->nx, pt->y, pt->z, pt->red, pt->green, pt->blue );
                        break;
                    }
                    case 5:
                    {
                        if (aBin)
                        {
                            WriteType(aFP, pt->nx);
                            WriteType(aFP, pt->ny);
                            WriteType(aFP, pt->nz);
                            WriteType(aFP, pt->red);
                            WriteType(aFP, pt->green);
                            WriteType(aFP, pt->blue);
                            WriteType(aFP, pt->alpha);
                        }
                        else
                            fprintf(aFP,"%.7f %.7f %.7f %.7f %.7f %.7f %d %d %d %d\n",  pt->x, pt->y, pt->z, pt->x, pt->y, pt->z, pt->red, pt->green, pt->blue, pt->alpha );
                        break;
                    }
                }
        }

        ElFclose(aFP);

        if ( glist!=NULL ) delete [] glist;
        if ( plist!=NULL ) delete plist;

        return EXIT_SUCCESS;
    }
#endif

/*Footer-MicMac-eLiSe-25/06/2007

 Ce logiciel est un programme informatique servant �  la mise en
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
 associés au chargement,  �  l'utilisation,  �  la modification et/ou au
 développement et �  la reproduction du logiciel par l'utilisateur étant
 donné sa spécificité de logiciel libre, qui peut le rendre complexe �
 manipuler et qui le réserve donc �  des développeurs et des professionnels
 avertis possédant  des  connaissances  informatiques approfondies.  Les
 utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
 logiciel �  leurs besoins dans des conditions permettant d'assurer la
 sécurité de leurs systèmes et ou de leurs données et, plus généralement,
 �  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

 Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
 pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
 termes.
 Footer-MicMac-eLiSe-25/06/2007*/


