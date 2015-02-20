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

int Mascarpone_main(int argc,char ** argv)
{
    string aNameFiles, aNamePly, aNameOut, aOut;
    vector<string> aVFiles;
    string filename;

    double aAngleMax = 90.f;

    ElInitArgMain
    (
    argc,argv,
    LArgMain()	<< EAM(aNameFiles),
    LArgMain()		<< EAM(aNamePly,"Ply",true)
                    << EAM(aNameOut,"Out",true)
                    << EAM(aAngleMax,"Max",true,"Max angle between surface and viewing direction (degree expected in ]0,90]) - default is 90")
    );

    printf ("Angle max: %3.1f\n", aAngleMax);
    aAngleMax *= PI/180.f;

    //Maillage
    cMesh myMesh(aNamePly);

    #ifdef _DEBUG
        printf ("nb vertex: %d - nb faces: %d\n", myMesh.getVertexNumber(), myMesh.getFacesNumber());

        for (int aK = 0; aK <myMesh.getVertexNumber();++aK)
        {
            Pt3dr pt = myMesh.getVertex(aK);
            printf ("vertex %ddd : %f %f %f\n",aK, pt.x ,pt.y, pt.z);
        }

        for (int aK = 0; aK <myMesh.getFacesNumber();++aK)
        {
            cTriangle *tri = myMesh.getTriangle(aK);
            vector <int> idxList;
            tri->getVertexesIndexes(idxList);
            if (idxList.size() ==  3)
                printf ("triangle %d : %d %d %d\n", aK, idxList[0], idxList[1], idxList[2]);
        }
    #endif

    ELISE_ASSERT (myMesh.getFacesNumber() < 65536, "big mesh!! label image overflow!!!");

    //Images maitresses
    int pos = (int) aNameFiles.find('#',0);
    while (pos>0)
    {
        aVFiles.push_back(aNameFiles.substr(0,pos));
        aNameFiles = aNameFiles.substr(pos+1, aNameFiles.size());
        pos = (int) aNameFiles.find('#',0);
    }
    aVFiles.push_back(aNameFiles);

    vector <cZBuf> aZBuffers;

    //Pour chaque image, identification des triangles visibles et affectation des attributs
    for (unsigned int aK=0; aK < aVFiles.size(); ++aK)
    {
        cZBuf aZBuffer;

        filename = aVFiles[aK];

        printf ("Reading %s\n", filename.c_str());

        aZBuffer.Nuage() = cElNuage3DMaille::FromFileIm(filename);

        aZBuffer.setSelfSz();
        aZBuffer.setMaxAngle(aAngleMax);

        printf ("BasculerUnMaillage\n" );
        aZBuffer.BasculerUnMaillage(myMesh);

        #ifdef _DEBUG
            string name = StdPrefix(filename) + "_zbuf.tif";

            aZBuffer.write(name);

            name.replace(name.end()-9, name.end(), "_label.tif");

            aZBuffer.writeImLabel(name);
        #endif

        aZBuffer.ComputeVisibleTrianglesIndexes();

        //On affecte les attributs aux triangles visibles (pour l'instant l'angle �  la normale)
        myMesh.setTrianglesAttribute(aK, aZBuffer.Nuage()->Cam()->DirVisee(), aZBuffer.getVisibleTrianglesIndexes());

        aZBuffers.push_back(aZBuffer);
    }

    //Pour chaque image, on choisit les triangles "convenables" parmi les triangles visibles
    for (unsigned int aK=0; aK < aVFiles.size(); ++aK)
    {
        //Graphe d'adjacence
        RGraph *g = new RGraph(aZBuffers[aK].getVisibleTrianglesIndexes().size(), myMesh.getEdgesNumber());

        Im2D_BIN mask = aZBuffers[aK].ComputeMask(aK, myMesh);

        if (aNameOut=="")
        {
            aOut = StdPrefix(aVFiles[aK]) + "_mask.tif";
        }
        else
        {
            char buf[100];
            sprintf(buf, "%i", aK);
            aOut = StdPrefix(aNameOut) + buf + ".tif";
        }

        printf ("Saving %s\n", aOut.c_str());
        Tiff_Im::CreateFromIm(mask, aOut);
        printf ("Done\n");

        //ponderation entre attache aux donnees et regularisation
        myMesh.setLambda(0.0001f);

        //on remplit le graphe d'adjacence
        vector <int> TriIdxInGraph;
        myMesh.setGraph(aK, *g, TriIdxInGraph, aZBuffers[aK].getVisibleTrianglesIndexes());

        float flow = g->maxflow();

        printf("Flow = %4.2f\n", flow);

        Im2D_BIN mask2 = aZBuffers[aK].ComputeMask(TriIdxInGraph, *g, myMesh);

        if (aNameOut=="")
        {
            aOut = StdPrefix(aVFiles[aK]) + "_mask_mcmf.tif";
        }
        else
        {
            char buf[100];
            sprintf(buf, "%i", aK);
            aOut = StdPrefix(aNameOut) + buf + "_mcmf.tif";
        }

        printf ("Saving %s\n", aOut.c_str());
        Tiff_Im::CreateFromIm(mask2, aOut);
        printf ("Done\n");

        delete g;
    }

    return EXIT_SUCCESS;
}

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


